import logging
import typing
from collections.abc import AsyncIterator

from agents import (
    AgentUpdatedStreamEvent,
    RawResponsesStreamEvent,
    RunItemStreamEvent,
    RunResultStreaming,
    StreamEvent,
)
from agents.items import (
    HandoffCallItem,
    HandoffOutputItem,
    MessageOutputItem,
    ReasoningItem,
    ToolCallItem,
    ToolCallOutputItem,
)
from openai.types.responses.response_audio_delta_event import ResponseAudioDeltaEvent
from openai.types.responses.response_audio_done_event import ResponseAudioDoneEvent
from openai.types.responses.response_audio_transcript_delta_event import (
    ResponseAudioTranscriptDeltaEvent,
)
from openai.types.responses.response_audio_transcript_done_event import (
    ResponseAudioTranscriptDoneEvent,
)
from openai.types.responses.response_code_interpreter_call_code_delta_event import (
    ResponseCodeInterpreterCallCodeDeltaEvent,
)
from openai.types.responses.response_code_interpreter_call_code_done_event import (
    ResponseCodeInterpreterCallCodeDoneEvent,
)
from openai.types.responses.response_code_interpreter_call_completed_event import (
    ResponseCodeInterpreterCallCompletedEvent,
)
from openai.types.responses.response_code_interpreter_call_in_progress_event import (
    ResponseCodeInterpreterCallInProgressEvent,
)
from openai.types.responses.response_code_interpreter_call_interpreting_event import (
    ResponseCodeInterpreterCallInterpretingEvent,
)
from openai.types.responses.response_completed_event import ResponseCompletedEvent
from openai.types.responses.response_content_part_added_event import (
    ResponseContentPartAddedEvent,
)
from openai.types.responses.response_content_part_done_event import (
    ResponseContentPartDoneEvent,
)
from openai.types.responses.response_created_event import ResponseCreatedEvent
from openai.types.responses.response_error_event import ResponseErrorEvent
from openai.types.responses.response_failed_event import ResponseFailedEvent
from openai.types.responses.response_file_search_call_completed_event import (
    ResponseFileSearchCallCompletedEvent,
)
from openai.types.responses.response_file_search_call_in_progress_event import (
    ResponseFileSearchCallInProgressEvent,
)
from openai.types.responses.response_file_search_call_searching_event import (
    ResponseFileSearchCallSearchingEvent,
)
from openai.types.responses.response_function_call_arguments_delta_event import (
    ResponseFunctionCallArgumentsDeltaEvent,
)
from openai.types.responses.response_function_call_arguments_done_event import (
    ResponseFunctionCallArgumentsDoneEvent,
)
from openai.types.responses.response_in_progress_event import ResponseInProgressEvent
from openai.types.responses.response_incomplete_event import ResponseIncompleteEvent
from openai.types.responses.response_output_item_added_event import (
    ResponseOutputItemAddedEvent,
)
from openai.types.responses.response_output_item_done_event import (
    ResponseOutputItemDoneEvent,
)
from openai.types.responses.response_reasoning_summary_part_added_event import (
    ResponseReasoningSummaryPartAddedEvent,
)
from openai.types.responses.response_reasoning_summary_part_done_event import (
    ResponseReasoningSummaryPartDoneEvent,
)
from openai.types.responses.response_reasoning_summary_text_delta_event import (
    ResponseReasoningSummaryTextDeltaEvent,
)
from openai.types.responses.response_reasoning_summary_text_done_event import (
    ResponseReasoningSummaryTextDoneEvent,
)
from openai.types.responses.response_refusal_delta_event import (
    ResponseRefusalDeltaEvent,
)
from openai.types.responses.response_refusal_done_event import ResponseRefusalDoneEvent
from openai.types.responses.response_text_annotation_delta_event import (
    ResponseTextAnnotationDeltaEvent,
)
from openai.types.responses.response_text_delta_event import ResponseTextDeltaEvent
from openai.types.responses.response_text_done_event import ResponseTextDoneEvent
from openai.types.responses.response_web_search_call_completed_event import (
    ResponseWebSearchCallCompletedEvent,
)
from openai.types.responses.response_web_search_call_in_progress_event import (
    ResponseWebSearchCallInProgressEvent,
)
from openai.types.responses.response_web_search_call_searching_event import (
    ResponseWebSearchCallSearchingEvent,
)

logger = logging.getLogger(__name__)


class OpenAIAgentsStreamHandler:
    def __init__(self, run_result_streaming: RunResultStreaming):
        self.run_result_streaming = run_result_streaming

    async def stream_events(self) -> AsyncIterator[StreamEvent]:
        async for event in self.run_result_streaming.stream_events():
            await self.__on_event(event)
            yield event

    async def run_until_done(self) -> None:
        async for event in self.stream_events():
            await self.__on_event(event)

    async def __on_event(self, event: StreamEvent):
        await self.on_event(event)

        if event.type == "agent_updated_stream_event":
            event = typing.cast(AgentUpdatedStreamEvent, event)
            await self.on_agent_updated_stream_event(event)

        elif event.type == "raw_response_event":
            event = typing.cast(RawResponsesStreamEvent, event)
            await self.on_raw_response_event(event)

            if event.data.type == "response.audio.delta":
                event.data = typing.cast(ResponseAudioDeltaEvent, event.data)
                await self.on_audio_delta(event.data)

            elif event.data.type == "response.audio.done":
                event.data = typing.cast(ResponseAudioDoneEvent, event.data)
                await self.on_audio_done(event.data)

            elif event.data.type == "response.audio.transcript.delta":
                event.data = typing.cast(ResponseAudioTranscriptDeltaEvent, event.data)
                await self.on_audio_transcript_delta(event.data)

            elif event.data.type == "response.audio.transcript.done":
                event.data = typing.cast(ResponseAudioTranscriptDoneEvent, event.data)
                await self.on_audio_transcript_done(event.data)

            elif event.data.type == "response.code_interpreter_call.code.delta":
                event.data = typing.cast(
                    ResponseCodeInterpreterCallCodeDeltaEvent, event.data
                )
                await self.on_code_interpreter_call_code_delta(event.data)

            elif event.data.type == "response.code_interpreter_call.code.done":
                event.data = typing.cast(
                    ResponseCodeInterpreterCallCodeDoneEvent, event.data
                )
                await self.on_code_interpreter_call_code_done(event.data)

            elif event.data.type == "response.code_interpreter_call.completed":
                event.data = typing.cast(
                    ResponseCodeInterpreterCallCompletedEvent, event.data
                )
                await self.on_code_interpreter_call_completed(event.data)

            elif event.data.type == "response.code_interpreter_call.in_progress":
                event.data = typing.cast(
                    ResponseCodeInterpreterCallInProgressEvent, event.data
                )
                await self.on_code_interpreter_call_in_progress(event.data)

            elif event.data.type == "response.code_interpreter_call.interpreting":
                event.data = typing.cast(
                    ResponseCodeInterpreterCallInterpretingEvent, event.data
                )
                await self.on_code_interpreter_call_interpreting(event.data)

            elif event.data.type == "response.completed":
                event.data = typing.cast(ResponseCompletedEvent, event.data)
                await self.on_response_completed(event.data)

            elif event.data.type == "response.content_part.added":
                event.data = typing.cast(ResponseContentPartAddedEvent, event.data)
                await self.on_content_part_added(event.data)

            elif event.data.type == "response.content_part.done":
                event.data = typing.cast(ResponseContentPartDoneEvent, event.data)
                await self.on_content_part_done(event.data)

            elif event.data.type == "response.created":
                event.data = typing.cast(ResponseCreatedEvent, event.data)
                await self.on_response_created(event.data)

            elif event.data.type == "error":
                event.data = typing.cast(ResponseErrorEvent, event.data)
                await self.on_error(event.data)

            elif event.data.type == "response.file_search_call.completed":
                event.data = typing.cast(
                    ResponseFileSearchCallCompletedEvent, event.data
                )
                await self.on_file_search_call_completed(event.data)

            elif event.data.type == "response.file_search_call.in_progress":
                event.data = typing.cast(
                    ResponseFileSearchCallInProgressEvent, event.data
                )
                await self.on_file_search_call_in_progress(event.data)

            elif event.data.type == "response.file_search_call.searching":
                event.data = typing.cast(
                    ResponseFileSearchCallSearchingEvent, event.data
                )
                await self.on_file_search_call_searching(event.data)

            elif event.data.type == "response.function_call_arguments.delta":
                event.data = typing.cast(
                    ResponseFunctionCallArgumentsDeltaEvent, event.data
                )
                await self.on_function_call_arguments_delta(event.data)

            elif event.data.type == "response.function_call_arguments.done":
                event.data = typing.cast(
                    ResponseFunctionCallArgumentsDoneEvent, event.data
                )
                await self.on_function_call_arguments_done(event.data)

            elif event.data.type == "response.in_progress":
                event.data = typing.cast(ResponseInProgressEvent, event.data)
                await self.on_response_in_progress(event.data)

            elif event.data.type == "response.failed":
                event.data = typing.cast(ResponseFailedEvent, event.data)
                await self.on_response_failed(event.data)

            elif event.data.type == "response.incomplete":
                event.data = typing.cast(ResponseIncompleteEvent, event.data)
                await self.on_response_incomplete(event.data)

            elif event.data.type == "response.output_item.added":
                event.data = typing.cast(ResponseOutputItemAddedEvent, event.data)
                await self.on_output_item_added(event.data)

            elif event.data.type == "response.output_item.done":
                event.data = typing.cast(ResponseOutputItemDoneEvent, event.data)
                await self.on_output_item_done(event.data)

            elif event.data.type == "response.reasoning_summary_part.added":
                event.data = typing.cast(
                    ResponseReasoningSummaryPartAddedEvent, event.data
                )
                await self.on_reasoning_summary_part_added(event.data)

            elif event.data.type == "response.reasoning_summary_part.done":
                event.data = typing.cast(
                    ResponseReasoningSummaryPartDoneEvent, event.data
                )
                await self.on_reasoning_summary_part_done(event.data)

            elif event.data.type == "response.reasoning_summary_text.delta":
                event.data = typing.cast(
                    ResponseReasoningSummaryTextDeltaEvent, event.data
                )
                await self.on_reasoning_summary_text_delta(event.data)

            elif event.data.type == "response.reasoning_summary_text.done":
                event.data = typing.cast(
                    ResponseReasoningSummaryTextDoneEvent, event.data
                )
                await self.on_reasoning_summary_text_done(event.data)

            elif event.data.type == "response.refusal.delta":
                event.data = typing.cast(ResponseRefusalDeltaEvent, event.data)
                await self.on_refusal_delta(event.data)

            elif event.data.type == "response.refusal.done":
                event.data = typing.cast(ResponseRefusalDoneEvent, event.data)
                await self.on_refusal_done(event.data)

            elif event.data.type == "response.output_text.annotation.added":
                event.data = typing.cast(ResponseTextAnnotationDeltaEvent, event.data)
                await self.on_output_text_annotation_added(event.data)

            elif event.data.type == "response.output_text.delta":
                event.data = typing.cast(ResponseTextDeltaEvent, event.data)
                await self.on_output_text_delta(event.data)

            elif event.data.type == "response.output_text.done":
                event.data = typing.cast(ResponseTextDoneEvent, event.data)
                await self.on_output_text_done(event.data)

            elif event.data.type == "response.web_search_call.completed":
                event.data = typing.cast(
                    ResponseWebSearchCallCompletedEvent, event.data
                )
                await self.on_web_search_call_completed(event.data)

            elif event.data.type == "response.web_search_call.in_progress":
                event.data = typing.cast(
                    ResponseWebSearchCallInProgressEvent, event.data
                )
                await self.on_web_search_call_in_progress(event.data)

            elif event.data.type == "response.web_search_call.searching":
                event.data = typing.cast(
                    ResponseWebSearchCallSearchingEvent, event.data
                )
                await self.on_web_search_call_searching(event.data)

            else:
                logger.warning(
                    f"Unhandled event.data.type: {event.data.type}: {event.data}"
                )

        elif event.type == "run_item_stream_event":
            event = typing.cast(RunItemStreamEvent, event)
            await self.on_run_item_stream_event(event)

            if event.item.type == "message_output_item":
                event.item = typing.cast(MessageOutputItem, event.item)
                await self.on_message_output_item(event.item)

            elif event.item.type == "handoff_call_item":
                event.item = typing.cast(HandoffCallItem, event.item)
                await self.on_handoff_call_item(event.item)

            elif event.item.type == "handoff_output_item":
                event.item = typing.cast(HandoffOutputItem, event.item)
                await self.on_handoff_output_item(event.item)

            elif event.item.type == "tool_call_item":
                event.item = typing.cast(ToolCallItem, event.item)
                await self.on_tool_call_item(event.item)

            elif event.item.type == "tool_call_output_item":
                event.item = typing.cast(ToolCallOutputItem, event.item)
                await self.on_tool_call_output_item(event.item)

            elif event.item.type == "reasoning_item":
                event.item = typing.cast(ReasoningItem, event.item)
                await self.on_reasoning_item(event.item)

            else:
                logger.warning(
                    f"Unhandled event.item.type: {event.item.type}: {event.item}"
                )

        else:
            logger.warning(f"Unhandled event type: {event.type}")

    async def on_event(self, event: StreamEvent) -> None:
        """Override this method to handle events."""
        pass

    async def on_agent_updated_stream_event(
        self, event: AgentUpdatedStreamEvent
    ) -> None:
        """Override this method to handle agent updated events."""
        pass

    async def on_raw_response_event(self, event: RawResponsesStreamEvent) -> None:
        """Override this method to handle raw response events."""
        pass

    async def on_run_item_stream_event(self, event: RunItemStreamEvent) -> None:
        """Override this method to handle run item stream events."""
        pass

    async def on_output_text_delta(self, event: ResponseTextDeltaEvent) -> None:
        """Override this method to handle output text delta events."""
        pass

    # Response Audio handlers
    async def on_audio_delta(self, event: ResponseAudioDeltaEvent) -> None:
        """Override this method to handle audio delta events."""
        pass

    async def on_audio_done(self, event: ResponseAudioDoneEvent) -> None:
        """Override this method to handle audio done events."""
        pass

    async def on_audio_transcript_delta(
        self, event: ResponseAudioTranscriptDeltaEvent
    ) -> None:
        """Override this method to handle audio transcript delta events."""
        pass

    async def on_audio_transcript_done(
        self, event: ResponseAudioTranscriptDoneEvent
    ) -> None:
        """Override this method to handle audio transcript done events."""
        pass

    # Code Interpreter handlers
    async def on_code_interpreter_call_code_delta(
        self, event: ResponseCodeInterpreterCallCodeDeltaEvent
    ) -> None:
        """Override this method to handle code interpreter call code delta events."""
        pass

    async def on_code_interpreter_call_code_done(
        self, event: ResponseCodeInterpreterCallCodeDoneEvent
    ) -> None:
        """Override this method to handle code interpreter call code done events."""
        pass

    async def on_code_interpreter_call_completed(
        self, event: ResponseCodeInterpreterCallCompletedEvent
    ) -> None:
        """Override this method to handle code interpreter call completed events."""
        pass

    async def on_code_interpreter_call_in_progress(
        self, event: ResponseCodeInterpreterCallInProgressEvent
    ) -> None:
        """Override this method to handle code interpreter call in progress events."""
        pass

    async def on_code_interpreter_call_interpreting(
        self, event: ResponseCodeInterpreterCallInterpretingEvent
    ) -> None:
        """Override this method to handle code interpreter call interpreting events."""
        pass

    # Response handlers
    async def on_response_completed(self, event: ResponseCompletedEvent) -> None:
        """Override this method to handle response completed events."""
        pass

    async def on_content_part_added(self, event: ResponseContentPartAddedEvent) -> None:
        """Override this method to handle content part added events."""
        pass

    async def on_content_part_done(self, event: ResponseContentPartDoneEvent) -> None:
        """Override this method to handle content part done events."""
        pass

    async def on_response_created(self, event: ResponseCreatedEvent) -> None:
        """Override this method to handle response created events."""
        pass

    async def on_error(self, event: ResponseErrorEvent) -> None:
        """Override this method to handle error events."""
        pass

    # File Search handlers
    async def on_file_search_call_completed(
        self, event: ResponseFileSearchCallCompletedEvent
    ) -> None:
        """Override this method to handle file search call completed events."""
        pass

    async def on_file_search_call_in_progress(
        self, event: ResponseFileSearchCallInProgressEvent
    ) -> None:
        """Override this method to handle file search call in progress events."""
        pass

    async def on_file_search_call_searching(
        self, event: ResponseFileSearchCallSearchingEvent
    ) -> None:
        """Override this method to handle file search call searching events."""
        pass

    # Function Call handlers
    async def on_function_call_arguments_delta(
        self, event: ResponseFunctionCallArgumentsDeltaEvent
    ) -> None:
        """Override this method to handle function call arguments delta events."""
        pass

    async def on_function_call_arguments_done(
        self, event: ResponseFunctionCallArgumentsDoneEvent
    ) -> None:
        """Override this method to handle function call arguments done events."""
        pass

    # More response handlers
    async def on_response_in_progress(self, event: ResponseInProgressEvent) -> None:
        """Override this method to handle response in progress events."""
        pass

    async def on_response_failed(self, event: ResponseFailedEvent) -> None:
        """Override this method to handle response failed events."""
        pass

    async def on_response_incomplete(self, event: ResponseIncompleteEvent) -> None:
        """Override this method to handle response incomplete events."""
        pass

    # Output item handlers
    async def on_output_item_added(self, event: ResponseOutputItemAddedEvent) -> None:
        """Override this method to handle output item added events."""
        pass

    async def on_output_item_done(self, event: ResponseOutputItemDoneEvent) -> None:
        """Override this method to handle output item done events."""
        pass

    # Reasoning summary handlers
    async def on_reasoning_summary_part_added(
        self, event: ResponseReasoningSummaryPartAddedEvent
    ) -> None:
        """Override this method to handle reasoning summary part added events."""
        pass

    async def on_reasoning_summary_part_done(
        self, event: ResponseReasoningSummaryPartDoneEvent
    ) -> None:
        """Override this method to handle reasoning summary part done events."""
        pass

    async def on_reasoning_summary_text_delta(
        self, event: ResponseReasoningSummaryTextDeltaEvent
    ) -> None:
        """Override this method to handle reasoning summary text delta events."""
        pass

    async def on_reasoning_summary_text_done(
        self, event: ResponseReasoningSummaryTextDoneEvent
    ) -> None:
        """Override this method to handle reasoning summary text done events."""
        pass

    # Refusal handlers
    async def on_refusal_delta(self, event: ResponseRefusalDeltaEvent) -> None:
        """Override this method to handle refusal delta events."""
        pass

    async def on_refusal_done(self, event: ResponseRefusalDoneEvent) -> None:
        """Override this method to handle refusal done events."""
        pass

    # Run item event handlers
    async def on_message_output_item(self, event: MessageOutputItem) -> None:
        """Override this method to handle message output item events."""
        pass

    async def on_handoff_call_item(self, event: HandoffCallItem) -> None:
        """Override this method to handle handoff call item events."""
        pass

    async def on_handoff_output_item(self, event: HandoffOutputItem) -> None:
        """Override this method to handle handoff output item events."""
        pass

    async def on_tool_call_item(self, event: ToolCallItem) -> None:
        """Override this method to handle tool call item events."""
        pass

    async def on_tool_call_output_item(self, event: ToolCallOutputItem) -> None:
        """Override this method to handle tool call output item events."""
        pass

    async def on_reasoning_item(self, event: ReasoningItem) -> None:
        """Override this method to handle reasoning item events."""
        pass

    async def on_output_text_annotation_added(
        self, event: ResponseTextAnnotationDeltaEvent
    ) -> None:
        """Override this method to handle output text annotation added events."""
        pass

    async def on_output_text_done(self, event: ResponseTextDoneEvent) -> None:
        """Override this method to handle output text done events."""
        pass

    async def on_web_search_call_completed(
        self, event: ResponseWebSearchCallCompletedEvent
    ) -> None:
        """Override this method to handle web search call completed events."""
        pass

    async def on_web_search_call_in_progress(
        self, event: ResponseWebSearchCallInProgressEvent
    ) -> None:
        """Override this method to handle web search call in progress events."""
        pass

    async def on_web_search_call_searching(
        self, event: ResponseWebSearchCallSearchingEvent
    ) -> None:
        """Override this method to handle web search call searching events."""
        pass
