import logging
from typing import AsyncIterator

from openai._streaming import AsyncStream
from openai.types.responses import ResponseStreamEvent
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


class OpenAIResponseStreamHandler:
    def __init__(self, stream: AsyncStream[ResponseStreamEvent]):
        self.stream = stream

    async def __aiter__(self) -> AsyncIterator[ResponseStreamEvent]:
        async for event in self.stream:
            await self.__on_event(event)
            yield event

    async def __anext__(self):
        return await self.stream.__anext__()

    async def __on_event(self, event: ResponseStreamEvent) -> None:
        await self.on_event(event)

        if event.type == "response.audio.delta":
            await self.on_response_audio_delta(event)
        elif event.type == "response.audio.done":
            await self.on_response_audio_done(event)
        elif event.type == "response.audio.transcript.delta":
            await self.on_response_audio_transcript_delta(event)
        elif event.type == "response.audio.transcript.done":
            await self.on_response_audio_transcript_done(event)
        elif event.type == "response.code_interpreter_call.code.delta":
            await self.on_response_code_interpreter_call_code_delta(event)
        elif event.type == "response.code_interpreter_call.code.done":
            await self.on_response_code_interpreter_call_code_done(event)
        elif event.type == "response.code_interpreter_call.completed":
            await self.on_response_code_interpreter_call_completed(event)
        elif event.type == "response.code_interpreter_call.in_progress":
            await self.on_response_code_interpreter_call_in_progress(event)
        elif event.type == "response.code_interpreter_call.interpreting":
            await self.on_response_code_interpreter_call_interpreting(event)
        elif event.type == "response.completed":
            await self.on_response_completed(event)
        elif event.type == "response.content_part.added":
            await self.on_response_content_part_added(event)
        elif event.type == "response.content_part.done":
            await self.on_response_content_part_done(event)
        elif event.type == "response.created":
            await self.on_response_created(event)
        elif event.type == "error":
            await self.on_response_error(event)
        elif event.type == "response.file_search_call.completed":
            await self.on_response_file_search_call_completed(event)
        elif event.type == "response.file_search_call.in_progress":
            await self.on_response_file_search_call_in_progress(event)
        elif event.type == "response.file_search_call.searching":
            await self.on_response_file_search_call_searching(event)
        elif event.type == "response.function_call_arguments.delta":
            await self.on_response_function_call_arguments_delta(event)
        elif event.type == "response.function_call_arguments.done":
            await self.on_response_function_call_arguments_done(event)
        elif event.type == "response.in_progress":
            await self.on_response_in_progress(event)
        elif event.type == "response.failed":
            await self.on_response_failed(event)
        elif event.type == "response.incomplete":
            await self.on_response_incomplete(event)
        elif event.type == "response.output_item.added":
            await self.on_response_output_item_added(event)
        elif event.type == "response.output_item.done":
            await self.on_response_output_item_done(event)
        elif event.type == "response.reasoning_summary_part.added":
            await self.on_response_reasoning_summary_part_added(event)
        elif event.type == "response.reasoning_summary_part.done":
            await self.on_response_reasoning_summary_part_done(event)
        elif event.type == "response.reasoning_summary_text.delta":
            await self.on_response_reasoning_summary_text_delta(event)
        elif event.type == "response.reasoning_summary_text.done":
            await self.on_response_reasoning_summary_text_done(event)
        elif event.type == "response.refusal.delta":
            await self.on_response_refusal_delta(event)
        elif event.type == "response.refusal.done":
            await self.on_response_refusal_done(event)
        elif event.type == "response.output_text.annotation.added":
            await self.on_response_output_text_annotation_added(event)
        elif event.type == "response.output_text.delta":
            await self.on_response_output_text_delta(event)
        elif event.type == "response.output_text.done":
            await self.on_response_output_text_done(event)
        elif event.type == "response.web_search_call.completed":
            await self.on_response_web_search_call_completed(event)
        elif event.type == "response.web_search_call.in_progress":
            await self.on_response_web_search_call_in_progress(event)
        elif event.type == "response.web_search_call.searching":
            await self.on_response_web_search_call_searching(event)
        else:
            logger.warning(f"Unhandled event type: {event.type}")

    async def on_event(self, event: ResponseStreamEvent):
        """Base handler for all events."""
        pass

    async def on_response_audio_delta(self, event: ResponseAudioDeltaEvent):
        """Handle audio delta events."""
        pass

    async def on_response_audio_done(self, event: ResponseAudioDoneEvent):
        """Handle audio done events."""
        pass

    async def on_response_audio_transcript_delta(
        self, event: ResponseAudioTranscriptDeltaEvent
    ):
        """Handle audio transcript delta events."""
        pass

    async def on_response_audio_transcript_done(
        self, event: ResponseAudioTranscriptDoneEvent
    ):
        """Handle audio transcript done events."""
        pass

    async def on_response_code_interpreter_call_code_delta(
        self, event: ResponseCodeInterpreterCallCodeDeltaEvent
    ):
        """Handle code interpreter call code delta events."""
        pass

    async def on_response_code_interpreter_call_code_done(
        self, event: ResponseCodeInterpreterCallCodeDoneEvent
    ):
        """Handle code interpreter call code done events."""
        pass

    async def on_response_code_interpreter_call_completed(
        self, event: ResponseCodeInterpreterCallCompletedEvent
    ):
        """Handle code interpreter call completed events."""
        pass

    async def on_response_code_interpreter_call_in_progress(
        self, event: ResponseCodeInterpreterCallInProgressEvent
    ):
        """Handle code interpreter call in progress events."""
        pass

    async def on_response_code_interpreter_call_interpreting(
        self, event: ResponseCodeInterpreterCallInterpretingEvent
    ):
        """Handle code interpreter call interpreting events."""
        pass

    async def on_response_completed(self, event: ResponseCompletedEvent):
        """Handle completed events."""
        pass

    async def on_response_content_part_added(
        self, event: ResponseContentPartAddedEvent
    ):
        """Handle content part added events."""
        pass

    async def on_response_content_part_done(self, event: ResponseContentPartDoneEvent):
        """Handle content part done events."""
        pass

    async def on_response_created(self, event: ResponseCreatedEvent):
        """Handle created events."""
        pass

    async def on_response_error(self, event: ResponseErrorEvent):
        """Handle error events."""
        pass

    async def on_response_file_search_call_completed(
        self, event: ResponseFileSearchCallCompletedEvent
    ):
        """Handle file search call completed events."""
        pass

    async def on_response_file_search_call_in_progress(
        self, event: ResponseFileSearchCallInProgressEvent
    ):
        """Handle file search call in progress events."""
        pass

    async def on_response_file_search_call_searching(
        self, event: ResponseFileSearchCallSearchingEvent
    ):
        """Handle file search call searching events."""
        pass

    async def on_response_function_call_arguments_delta(
        self, event: ResponseFunctionCallArgumentsDeltaEvent
    ):
        """Handle function call arguments delta events."""
        pass

    async def on_response_function_call_arguments_done(
        self, event: ResponseFunctionCallArgumentsDoneEvent
    ):
        """Handle function call arguments done events."""
        pass

    async def on_response_in_progress(self, event: ResponseInProgressEvent):
        """Handle in progress events."""
        pass

    async def on_response_failed(self, event: ResponseFailedEvent):
        """Handle failed events."""
        pass

    async def on_response_incomplete(self, event: ResponseIncompleteEvent):
        """Handle incomplete events."""
        pass

    async def on_response_output_item_added(self, event: ResponseOutputItemAddedEvent):
        """Handle output item added events."""
        pass

    async def on_response_output_item_done(self, event: ResponseOutputItemDoneEvent):
        """Handle output item done events."""
        pass

    async def on_response_reasoning_summary_part_added(
        self, event: ResponseReasoningSummaryPartAddedEvent
    ):
        """Handle reasoning summary part added events."""
        pass

    async def on_response_reasoning_summary_part_done(
        self, event: ResponseReasoningSummaryPartDoneEvent
    ):
        """Handle reasoning summary part done events."""
        pass

    async def on_response_reasoning_summary_text_delta(
        self, event: ResponseReasoningSummaryTextDeltaEvent
    ):
        """Handle reasoning summary text delta events."""
        pass

    async def on_response_reasoning_summary_text_done(
        self, event: ResponseReasoningSummaryTextDoneEvent
    ):
        """Handle reasoning summary text done events."""
        pass

    async def on_response_refusal_delta(self, event: ResponseRefusalDeltaEvent):
        """Handle refusal delta events."""
        pass

    async def on_response_refusal_done(self, event: ResponseRefusalDoneEvent):
        """Handle refusal done events."""
        pass

    async def on_response_output_text_annotation_added(
        self, event: ResponseTextAnnotationDeltaEvent
    ):
        """Handle output text annotation added events."""
        pass

    async def on_response_output_text_delta(self, event: ResponseTextDeltaEvent):
        """Handle output text delta events."""
        pass

    async def on_response_output_text_done(self, event: ResponseTextDoneEvent):
        """Handle output text done events."""
        pass

    async def on_response_web_search_call_completed(
        self, event: ResponseWebSearchCallCompletedEvent
    ):
        """Handle web search call completed events."""
        pass

    async def on_response_web_search_call_in_progress(
        self, event: ResponseWebSearchCallInProgressEvent
    ):
        """Handle web search call in progress events."""
        pass

    async def on_response_web_search_call_searching(
        self, event: ResponseWebSearchCallSearchingEvent
    ):
        """Handle web search call searching events."""
        pass
