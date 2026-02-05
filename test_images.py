import asyncio, os, tempfile, importlib, base64
from unittest.mock import AsyncMock, MagicMock


async def main():
    with tempfile.TemporaryDirectory() as d:
        os.environ['DATA_DIR'] = d
        import src.config
        importlib.reload(src.config)

        from src.tools.registry import TOOL_REGISTRY, ToolContext
        import src.tools.definitions  # noqa
        from src.tools.executor import ToolExecutor
        from src.agent.agent import Agent, OpenAICompatibleClient, AgentResponse, AgentTextBlock

        ctx = ToolContext()
        ex = ToolExecutor(registry=TOOL_REGISTRY, ctx=ctx)
        await ex.initialize()

        agent = Agent(model_api='anthropic', model_name='x', model_api_key='d', tool_executor=ex)
        captured = []

        async def fake_complete(**kwargs):
            captured.append(kwargs.get('messages'))
            return AgentResponse(content=[AgentTextBlock(text='ok')], stop_reason='end_turn', usage=None)

        agent._client = MagicMock()
        agent._client.complete = fake_complete

        b64 = base64.b64encode(b'fake-png-bytes').decode('ascii')

        # 1. multi-block user message with image first, text last
        await agent.chat('what is this?', images=[{'media_type': 'image/png', 'data': b64}])
        user_msg = captured[-1][-1]
        assert isinstance(user_msg['content'], list)
        assert user_msg['content'][0]['type'] == 'image'
        assert user_msg['content'][0]['source']['media_type'] == 'image/png'
        assert user_msg['content'][1]['type'] == 'text'
        print('pass 1: multi-block built (image first, text last)')

        # 2. plain string when no images
        await agent.chat('just text')
        assert isinstance(captured[-1][-1]['content'], str)
        print('pass 2: plain string when no images')

        # 3. OpenAI-compat client converts image blocks
        oai = OpenAICompatibleClient(api_key='x', model_name='g', endpoint='http://localhost:11434/v1')
        conv = [{
            'role': 'user',
            'content': [
                {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/jpeg', 'data': b64}},
                {'type': 'text', 'text': 'describe'},
            ],
        }]
        conv_out = oai._convert_messages(conv, 'sys')
        parts = conv_out[1]['content']
        assert parts[0]['type'] == 'image_url'
        assert parts[0]['image_url']['url'].startswith('data:image/jpeg;base64,')
        assert parts[1]['type'] == 'text' and parts[1]['text'] == 'describe'
        print('pass 3: OAI-compat converts image blocks to data URL')

        # 4. plain string user still works in OAI-compat
        conv_plain = [{'role': 'user', 'content': 'hello'}]
        assert oai._convert_messages(conv_plain, 'sys')[1] == {'role': 'user', 'content': 'hello'}
        print('pass 4: plain-string user unchanged')

        # 5. Discord _extract_images filters correctly
        from src.discord_bot.bot import _extract_images, MAX_IMAGE_BYTES
        fake = MagicMock()
        good = MagicMock()
        good.content_type = 'image/png'
        good.size = 1024
        good.filename = 'a.png'
        good.read = AsyncMock(return_value=b'bytes')
        big = MagicMock()
        big.content_type = 'image/jpeg'
        big.size = MAX_IMAGE_BYTES + 1
        big.filename = 'b.jpg'
        big.read = AsyncMock()
        txt = MagicMock()
        txt.content_type = 'text/plain'
        txt.size = 100
        txt.filename = 'c.txt'
        txt.read = AsyncMock()
        fake.attachments = [good, big, txt]
        out = await _extract_images(fake)
        assert len(out) == 1
        assert out[0]['media_type'] == 'image/png'
        assert out[0]['data'] == base64.b64encode(b'bytes').decode('ascii')
        print('pass 5: _extract_images filters non-image and oversized')

        await ctx._store.close()
        print('ALL PASS')


asyncio.run(main())
