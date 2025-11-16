from multiprocessing.managers import BaseManager
import time

class ModelClient(BaseManager): pass
ModelClient.register("get_model")

manager = ModelClient(address=("localhost", 7002), authkey=b"moe_model_key")
manager.connect()

model = manager.get_model()

user_instruction = "Draw a space station orbiting Saturn"

start = time.time()
raw = model.fast_inference(user_instruction)
end = time.time()

print(raw)
print("Time:", end - start)
