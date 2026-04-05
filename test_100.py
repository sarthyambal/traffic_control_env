import asyncio
import time
import uuid

from models import TrafficControlAction
from server.traffic_control_env_environment import TrafficControlEnvironment

async def run_query(query_id: int):
    # Native execution bypassing HTTP by providing environment_class directly
    env = TrafficControlEnvironment()
    
    # 1. Reset
    result = env.reset(scenario_id="easy_rush_hour", episode_id=f"test_ep_{query_id}_{uuid.uuid4().hex[:8]}")
    
    # 2. Step 1 Tool
    action1 = TrafficControlAction(tool_name="get_intersection_status", tool_args={"intersection_id": "INT-1"})
    res1 = env.step(action1)
    
    # 3. Step 2 Action
    action2 = TrafficControlAction(tool_name="set_traffic_light", tool_args={"intersection_id": "INT-1", "phase": "NS_GREEN"})
    res2 = env.step(action2)
    
    # 4. Message
    action3 = TrafficControlAction(message="cleared green north south")
    res3 = env.step(action3)
    
    return res3.reward or 0.0

async def main():
    print("Starting 100 concurrent queries against TrafficControlEnvironment natively...")
    start_time = time.time()
    
    # Fire off 100 concurrent evaluate workflows
    tasks = [asyncio.create_task(run_query(i)) for i in range(100)]
    results = await asyncio.gather(*tasks)
    
    duration = time.time() - start_time
    
    success_count = sum(1 for r in results if r > 0.0)
    print(f"Finished 100 queries in {duration:.2f} seconds!")
    print(f"Successful resolutions: {success_count} / 100")
    
if __name__ == "__main__":
    asyncio.run(main())
