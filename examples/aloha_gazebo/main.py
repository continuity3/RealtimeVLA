"""
Main script for running ALOHA policy in Gazebo simulation.

This script connects to a policy server and runs the policy in Gazebo.
"""
import dataclasses
import logging
import pathlib

import env as _env
from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
import tyro


@dataclasses.dataclass
class Args:
    """Command line arguments."""
    
    # Policy server configuration
    host: str = "localhost"
    port: int = 8000
    
    # Action horizon for action chunking
    action_horizon: int = 10
    
    # ROS2 topic names (adjust if your Gazebo setup uses different topics)
    image_topic_high: str = "/camera_high/image_raw"
    image_topic_low: str = "/camera_low/image_raw"
    image_topic_left_wrist: str = "/camera_left_wrist/image_raw"
    image_topic_right_wrist: str = "/camera_right_wrist/image_raw"
    state_topic: str = "/aloha/joint_states"
    action_topic: str = "/aloha/joint_commands"
    
    # Image resolution
    render_height: int = 224
    render_width: int = 224
    
    # Runtime configuration
    max_hz: float = 50.0


def main(args: Args) -> None:
    """Main function."""
    logging.basicConfig(level=logging.INFO)
    
    # Create environment
    environment = _env.GazeboAlohaEnvironment(
        image_topic_high=args.image_topic_high,
        image_topic_low=args.image_topic_low,
        image_topic_left_wrist=args.image_topic_left_wrist,
        image_topic_right_wrist=args.image_topic_right_wrist,
        state_topic=args.state_topic,
        action_topic=args.action_topic,
        render_height=args.render_height,
        render_width=args.render_width,
    )
    
    # Create policy agent
    agent = _policy_agent.PolicyAgent(
        policy=action_chunk_broker.ActionChunkBroker(
            policy=_websocket_client_policy.WebsocketClientPolicy(
                host=args.host,
                port=args.port,
            ),
            action_horizon=args.action_horizon,
        )
    )
    
    # Create runtime
    runtime = _runtime.Runtime(
        environment=environment,
        agent=agent,
        max_hz=args.max_hz,
    )
    
    try:
        runtime.run()
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    finally:
        environment.shutdown()


if __name__ == "__main__":
    tyro.cli(main)

