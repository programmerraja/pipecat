import asyncio

class Frame:
    def __init__(self, data):
        self.data = data

class FrameProcessor:
    def __init__(self, name):
        self.name = name
        self._next = None
        self._input_queue = asyncio.Queue()
        self._task = None

    def link(self, next_processor):
        self._next = next_processor

    async def start(self):
        self._task = asyncio.create_task(self._run())

    async def stop(self):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _run(self):
        while True:
            frame = await self._input_queue.get()
            await self.process_frame(frame)
            self._input_queue.task_done()

    async def queue_frame(self, frame):
        await self._input_queue.put(frame)

    async def push_frame(self, frame):
        if self._next:
            await self._next.queue_frame(frame)

    async def process_frame(self, frame):
        # Default: just pass the frame to the next processor
        print(f"{self.name}: Passing frame")
        await self.push_frame(frame)

# Example: STT Processor
class STTProcessor(FrameProcessor):
    async def process_frame(self, frame):
        print(f"{self.name}: Received audio '{frame.data}'")
        # Fake speech-to-text
        text = frame.data.upper()  # Just uppercase for demo
        new_frame = Frame(text)
        print(f"{self.name}: Transcribed to '{text}'")
        await self.push_frame(new_frame)

# Example: TTS Processor
class TTSProcessor(FrameProcessor):
    async def process_frame(self, frame):
        print(f"{self.name}: Received text '{frame.data}'")
        # Fake text-to-speech
        audio = f"audio({frame.data})"
        print(f"{self.name}: Synthesized '{audio}'")
        await self.push_frame(Frame(audio))

# Pipeline to link processors
class Pipeline:
    def __init__(self, processors):
        self.processors = processors
        # Link them in order
        for i in range(len(processors) - 1):
            processors[i].link(processors[i+1])

    async def start(self):
        for p in self.processors:
            await p.start()

    async def stop(self):
        for p in self.processors:
            await p.stop()

    async def push(self, frame):
        await self.processors[0].queue_frame(frame)
