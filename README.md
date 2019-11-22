
*** WE HAVE MOVED TO https://gitlab.com/mad_rl_ ***


MAD_RL_ SFII Challenge
===


- ROM name = `'Street Fighter II' - Special Champion Edition (USA).md`

- Agent base model = Asynchronous Advantage Actor-Critic (A3C)  


## How to test it

1. Download SFII Challenge Engine docker:
```
docker pull madrl/sfii-challenge-engine:0.3.2
```

2. Execute agent with the engine docker:
```
docker run -v $PWD/roms:/roms/ -v $PWD/models/:/models/ -v $PWD/:/requirements -v $PWD/sfii_agent_base/:/sfii_agent_base/ -e AGENT_MODULE=sfii_agent_base.agent madrl/sfii-challenge-engine:0.3.2
```


## How to play with your agent

You can modify some parameters using environment variables:


```
ENGINE_PARAMETERS 

    "EPISODES_TRAINING" = 10,
    "EPISODES_TESTING"= 5,
    "NUM_PROCESSES" = 5,
    "OUTPUT_MODELS_PATH" = "models",
    "DELAY_FRAMES" = 50,
    "ENGINE_MODULE" = "src.environments.gym_retro.engine",
    "ENGINE_CLASS" = "Engine"
```

Default agent parameters that you can modify inside the agent.py:

```
AGENT_PARAMETERS

    'frames': 16,
    'cnn_channels': 32,
    'n_outputs': 49,
    'screen_height': 256,
    'screen_width': 200,
    'width': 80,
    'height': 80,
    'start_from_model': "models/sf2_a3c.pth",
    'module': "src.environments.gym_retro.my_agent.agent",
    "class": "Agent"

```
