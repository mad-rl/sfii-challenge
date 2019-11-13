MAD_RL_ SFII Challenge
===
 

- ROM name = `Street Fighter II' - Special Champion Edition (USA).md`

- SFII Challenge Engine docker name = `madrl/sfii-challenge-engine:v0.3.1`

- Agent base model = A3C 


## How to test it

```
docker run -v $PWD/roms:/roms/ -v $PWD/models/:/models/ -v $PWD/sfii_agent_base/:/sfii_agent_base/ -e AGENT_MODULE=sfii_agent_base.agent madrl/sfii-challenge-engine:v0.3.1
```


## How to play with your agent

You can modify some parameters using environment variables


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