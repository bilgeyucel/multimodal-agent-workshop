from typing import AsyncGenerator, Annotated, Optional, List, Literal

from haystack import Pipeline
from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage, ImageContent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.generators.utils import print_streaming_chunk
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.converters import MarkdownToDocument
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.tools import tool, PipelineTool

from haystack_experimental.chat_message_stores.in_memory import InMemoryChatMessageStore
from haystack_experimental.components.retrievers import ChatMessageRetriever
from haystack_experimental.components.writers import ChatMessageWriter

from hayhooks import BasePipelineWrapper, async_streaming_generator
from hayhooks.open_webui import OpenWebUIEvent, create_notification_event, create_status_event, create_details_tag


class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        # Document store and preprocessing pipeline
        document_store = InMemoryDocumentStore()
        markdown_converter = MarkdownToDocument()
        document_splitter = DocumentSplitter(split_by="word", split_length=150, split_overlap=50)
        document_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
        document_writer = DocumentWriter(document_store)

        preprocessing_pipeline = Pipeline()
        preprocessing_pipeline.add_component(instance=markdown_converter, name="markdown_converter")
        preprocessing_pipeline.add_component(instance=document_splitter, name="document_splitter")
        preprocessing_pipeline.add_component(instance=document_embedder, name="document_embedder")
        preprocessing_pipeline.add_component(instance=document_writer, name="document_writer")

        preprocessing_pipeline.connect("markdown_converter", "document_splitter")
        preprocessing_pipeline.connect("document_splitter", "document_embedder")
        preprocessing_pipeline.connect("document_embedder", "document_writer")

        # Run preprocessing pipeline on the policy document
        preprocessing_pipeline.run({"markdown_converter": {"sources": ["files/social_budget_policy.md"]}})

        # Retrieval pipeline
        text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
        retriever = InMemoryEmbeddingRetriever(document_store)

        retrieval_pipeline = Pipeline()
        retrieval_pipeline.add_component("text_embedder", text_embedder)
        retrieval_pipeline.add_component("retriever", retriever)
        retrieval_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

        # Retrieval tool
        def doc_to_string(documents) -> str:
            result_str = ""
            for document in documents:
                result_str += f"File Content for {document.meta['file_path']}: {document.content}\n\n"
            return result_str

        retrieval_tool = PipelineTool(
            pipeline=retrieval_pipeline,
            input_mapping={"query": ["text_embedder.text"]},
            output_mapping={"retriever.documents": "documents"},
            name="retrieval_tool",
            description="Search information for company policies at deepset",
            outputs_to_string={"source": "documents", "handler": doc_to_string},
        )

        # Send reimbursement tool
        @tool
        def send_reimbursement(
            amount: Annotated[float, "the exact amount of the spend"],
            currency: Annotated[Literal["eur", "usd"], "the currency of the spend"],
            description: Annotated[str, "the description of the spend"]
        ) -> str:
            '''
            A simple tool to send the reimbursement requests with amount, currency and description details
            '''
            if amount > 10:
                return "This amount cannot be reimbursed"
            return "We received your request ;)"

        # System prompt
        system_prompt = """
        You are a helpful deepset agent.
        Your users are deepset employees, trying to understand the company policies.
        You have access to a list of tools:
        - `retrieval_tool`: to get information on deepset policies like social budget, business trip expenses etc.
        - `send_reimbursement`: to submit reimbursement requests to the internal finance tool
        Use relevant tools to address the user query if necessary.
        Consider all information (image, textual context, previous messages etc.) you have to address the user query.
        Share your reasoning with the user with the final statement.
        Ask for more info if you don't have enough details to answer a question. Don't make assumptions without relevant info.
        Give suggestions or solutions for users questions if possible.
        """

        # Generator
        generator = OpenAIChatGenerator(model="gpt-4o")

        # Agent Component
        agent = Agent(
            system_prompt=system_prompt,
            chat_generator=generator,
            tools=[retrieval_tool, send_reimbursement],
            streaming_callback=print_streaming_chunk
        )

        # Chat message store for conversational memory
        message_store = InMemoryChatMessageStore()

        # Conversational agent pipeline
        self.conversational_agent = Pipeline()
        self.conversational_agent.add_component("agent", agent)
        self.conversational_agent.add_component("message_retriever", ChatMessageRetriever(message_store))
        self.conversational_agent.add_component("message_writer", ChatMessageWriter(message_store))

        # Connections for the pipeline
        self.conversational_agent.connect("message_retriever.messages", "agent.messages")
        self.conversational_agent.connect("agent.messages", "message_writer")
    
    # Deploy Tool Calling Agent 
    def run_api(self, query: str, image_path: Optional[str] = None, chat_history_id: Optional[str] = "default") -> str:
        if image_path:
            image = ImageContent.from_file_path(image_path)
            content_parts = [query, image]
        else:
            content_parts = [query]
        
        result = self.conversational_agent.run(
            data={
                "message_retriever": {
                    "current_messages": [ChatMessage.from_user(content_parts=content_parts)],
                    "chat_history_id": chat_history_id,
                },
                "message_writer": {"chat_history_id": chat_history_id},
            }
        )
        return result["agent"]["last_message"].text
    
    # Open WebUI hooks
    def on_tool_call_start(
        self, tool_name: str, arguments: dict, id: str
    ) -> List[OpenWebUIEvent]:
        return [
            create_status_event(description=f"Tool call started: {tool_name}"),
            create_notification_event(
                notification_type="info",
                content=f"Tool call started: {tool_name}",
            )
        ]
    
    def on_tool_call_end(
        self,
        tool_name: str,
        arguments: dict,
        result: str,
        error: bool,  # noqa: ARG002
    ) -> list[OpenWebUIEvent]:
        return [
            create_status_event(
                description=f"Tool call ended: {tool_name} with arguments: {arguments}",
                done=True,
            ),
            create_notification_event(
                notification_type="success",
                content=f"Tool call ended: {tool_name}",
            ),
            create_details_tag(
                tool_name=tool_name,
                summary=f"Tool call result for {tool_name}",
                content=(f"```\nArguments:\n{arguments}\n\nResponse:\n{result}\n```"),
            ),
        ]
    
    # handles OpenAI-compatible chat completion requests asynchronously for Open WebUI
    async def run_chat_completion_async(
        self, model: str, messages: list[dict], body: dict
    ) -> AsyncGenerator[str, None]:
        chat_messages = [
            ChatMessage.from_openai_dict_format(message) for message in messages
        ]
        # Extract chat_history_id from body if available, otherwise use default
        chat_history_id = body.get("chat_history_id", "default")
        
        return async_streaming_generator(
            on_tool_call_start=self.on_tool_call_start,
            on_tool_call_end=self.on_tool_call_end,
            pipeline=self.conversational_agent,
            pipeline_run_args={
                "message_retriever": {
                    "current_messages": chat_messages,
                    "chat_history_id": chat_history_id,
                },
                "message_writer": {"chat_history_id": chat_history_id},
            },
        )