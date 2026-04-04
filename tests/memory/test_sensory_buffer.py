from brain_agent.memory.sensory_buffer import SensoryBuffer

def test_register_and_get():
    buf = SensoryBuffer()
    buf.register({"text": "hello"}, modality="text")
    items = buf.get_all()
    assert len(items) == 1
    assert items[0].data["text"] == "hello"

def test_flush_clears_buffer():
    buf = SensoryBuffer()
    buf.register({"text": "hello"}, modality="text")
    buf.flush()
    assert len(buf.get_all()) == 0

def test_attend_filters_items():
    buf = SensoryBuffer()
    buf.register({"text": "important", "priority": 0.9}, modality="text")
    buf.register({"text": "noise", "priority": 0.1}, modality="text")
    attended = buf.attend(lambda item: item.data.get("priority", 0) > 0.5)
    assert len(attended) == 1
    assert attended[0].data["text"] == "important"

def test_new_request_flushes_previous():
    buf = SensoryBuffer()
    buf.register({"text": "old"}, modality="text")
    buf.new_cycle()
    buf.register({"text": "new"}, modality="text")
    assert len(buf.get_all()) == 1
    assert buf.get_all()[0].data["text"] == "new"

def test_unlimited_capacity():
    buf = SensoryBuffer()
    for i in range(1000):
        buf.register({"i": i}, modality="text")
    assert len(buf.get_all()) == 1000
