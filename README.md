# procgen-PPO
Project of the course Autonomous and Adaptive Systems from Unibo, 2024.

## Requirements
To install the required packages:

```
pip list --format=freeze > requirements.txt
```

## Training

To train the agent on the cloud (usually with Colab or Kaggle), move the file ``notebook.ipynb`` to the chosen platform and run it with the desired configuration.

To train the agent using "local" resources, modify the configuration dictionary in main.py with the desired game, difficulty and hyperparameters.
Then run:

```
python main.py 
```

I didn't provide the game and difficulty settings as command line parameters as training has been done using cloud GPU resources.

## Demo
To run a demo of the trained models, use the command:

```
python demo.py --game <game> --difficulty <difficulty>
```

Supported (game, difficulty) configurations are:

* ``(coinrun, hard)``
* ``(miner, easy)``
* ``(bossfight, easy)``
* ``(climber, easy)``
