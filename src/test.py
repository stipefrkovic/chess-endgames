def td_lambda_gradient_test():
    import tensorflow as tf

    x = tf.Variable(3.0)

    with tf.GradientTape() as tape:
        y = x ** 2
        y *= 2

    dy_dx = tape.gradient(y, x)
    print(dy_dx.numpy())


def td_lambda_gradient_test_2():
    import tensorflow as tf

    layer = tf.keras.layers.Dense(4, activation='tanh')
    x = tf.constant([[2.]])

    # print("Layer weights: " + str(layer.get_weights()))
    print("Input: " + str(x.numpy()))

    with tf.GradientTape() as tape:
        y = layer(x)
        print("Output: " + str(y))
        # loss = tf.reduce_mean(y ** 2)
        # print("Loss: " + str(loss))

    print("Layer weights: " + str(layer.get_weights()))

    dy_dx = tape.gradient(y, layer.trainable_weights)
    print("Gradients y:" + str(dy_dx))

    with tf.GradientTape() as tape:
        y = layer(x)
        print("Output: " + str(y))
        # loss = tf.reduce_mean(y ** 2)
        # print("Loss: " + str(loss))
        z = y * 2

    print("Layer weights: " + str(layer.get_weights()))

    dz_dx = tape.gradient(z, layer.trainable_weights)
    print("Gradients z:" + str(dz_dx))

    optimizer = tf.keras.optimizers.SGD(learning_rate=1)
    optimizer.apply_gradients(zip(dy_dx, layer.trainable_weights))
    y = layer(x)
    print("Output: " + str(y))

def queen_abs_evaluation():
    from game import QueenEndgameGame
    queen_endgame_game = QueenEndgameGame(
        max_depth=5,
        train=True,
        train_steps=10,
        lambda_value=0.9,
        name="queen_endgame_raw_train"
    )
    queen_endgame_game.plot_reward_losses()

def queen_abs_evaluation_2():
    from game import QueenEndgameGame
    queen_endgame_game = QueenEndgameGame(
        max_depth=5,
        train=True,
        train_steps=10,
        lambda_value=0.9,
        name="queen_endgame_raw_train"
    )
    queen_endgame_game.plot_old_new_state_losses('lambda_td')

def plot_reward_losses():
    from pathlib import Path
    from matplotlib import pyplot as plt
    from matplotlib.transforms import Affine2D
    from matplotlib.ticker import FormatStrFormatter
    from itertools import chain
    import numpy as np
    import pandas as pd
    from logger import logger
    plt.rcParams.update({'font.size': 12})
    
    results_path = str(Path().absolute()) + '/src/results/'
    figures_path = str(Path().absolute()) + '/src/figures/'
    
    raw_name = "rook_endgame_raw_train"
    raw_df = pd.read_table(f"{results_path}{raw_name}_reward_loss.csv", sep=",", header=None)
    raw_reward_loss_2d = np.array(raw_df)
    raw_reward_losses_1d = list(chain(*raw_reward_loss_2d))
    raw_abs_reward_losses_1d = [abs(ele) for ele in raw_reward_losses_1d]
    logger.info(f"raw_reward_loss_mean: {np.mean(raw_abs_reward_losses_1d)}")

    transfer_name = "rook_endgame_transfer_train"
    transfer_df = pd.read_table(f"{results_path}{transfer_name}_reward_loss.csv", sep=",", header=None)
    transfer_reward_loss_2d = np.array(transfer_df)
    transfer_reward_losses_1d = list(chain(*transfer_reward_loss_2d))
    transfer_abs_reward_losses_1d = [abs(ele) for ele in transfer_reward_losses_1d]
    logger.info(f"raw_reward_loss_mean: {np.mean(transfer_abs_reward_losses_1d)}")

    logger.info("Plotting reward losses")
    fig, ax = plt.subplots()
    plt.xlabel('Game')
    plt.ylabel('Absolute difference between\nminimax evaluation and true evaluation')
    plt.axhline(0, color='black', linestyle='dashed', linewidth=1)
    # plt.xticks(range(1, len(reward_losses_1d)+1, 1))
    # plt.ylim(min(reward_losses_1d) - 0.1, max(reward_losses_1d) + 0.1)
    plt.tight_layout()
    ax.plot(range(1, len(raw_abs_reward_losses_1d)+1, 1), raw_abs_reward_losses_1d, label="Random initial weights")
    ax.plot(range(1, len(transfer_abs_reward_losses_1d)+1, 1), transfer_abs_reward_losses_1d, label="Pre-trained weights")
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax.legend()
    fig.savefig(f"{figures_path}baseline_transfer_abs_evaluation_loss.png")


def main():
    queen_abs_evaluation()
    queen_abs_evaluation_2()
    plot_reward_losses()

if __name__ == "__main__":
    main()
