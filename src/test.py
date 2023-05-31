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


def main():
    import chess
    board = chess.Board().empty()
    board.set_piece_at(chess.F7, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.H7, chess.Piece(chess.QUEEN, chess.WHITE))
    # board.set_piece_at(chess.F5, chess.Piece(chess.QUEEN, chess.WHITE))
    board.set_piece_at(chess.H6, chess.Piece(chess.KING, chess.BLACK))
    board.turn = chess.BLACK
    print(board.is_valid())
    print(board.outcome())
    print(board.legal_moves)
    for legal_move in board.legal_moves:
        if board.is_capture(legal_move):
            print('stalemate')


if __name__ == "__main__":
    main()
