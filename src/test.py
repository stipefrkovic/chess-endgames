import chess
import numpy as np
from skimage.segmentation import flood, flood_fill
import math
import random
from itertools import chain

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


# def create_random_rook_checkmate():
#         random_rook = random.choice([chess.Piece(chess.ROOK, chess.WHITE),
#                                      chess.Piece(chess.ROOK, chess.BLACK)])
#         turn = random_rook.color
#         pieces = (chess.Piece(chess.KING, turn), 
#                   random_rook,
#                   chess.Piece(chess.KING, not turn),
#                   )
#         while True:
#             board = chess.Board().empty()
#             king = (random.choice(CHESS_FILES), random.choice(CHESS_RANKS))
#             board.set_piece_at(chess.square(king[0], king[1]), pieces[0])
#             king_neighbors = get_neighbor_squares(king)
#             piece = random.choice(king_neighbors)
#             board.set_piece_at(chess.square(piece[0], piece[1]), pieces[1])
#             break
            
#             board.turn = turn
#             if board.is_valid():
#                 break
#         return board  


def get_neighbor_squares(piece):
    file = piece[0]
    rank = piece[1]
    num_files = 8
    num_ranks = 8
    neighbor_squares = []
    # Check bottom-left neighbor
    if rank > 0 and file > 0:
        neighbor_squares.append((file-1, rank-1))
    # Check bottom neighbor
    if rank > 0:
        neighbor_squares.append((file, rank-1))
    # Check bottom-right neighbor
    if rank > 0 and file < num_files - 1:
        neighbor_squares.append((file+1, rank-1))
    # Check left neighbor
    if file > 0:
        neighbor_squares.append((file-1, rank))
    # Check right neighbor
    if file < num_files - 1:
        neighbor_squares.append((file+1, rank))
    # Check top-left neighbor
    if rank < num_ranks - 1 and file > 0:
        neighbor_squares.append((file-1, rank+1))
    # Check top neighbor
    if rank < num_ranks - 1:
        neighbor_squares.append((file, rank+1))
    # Check top-right neighbor
    if rank < num_ranks - 1 and file < num_files - 1:
        neighbor_squares.append((file+1, rank+1))
    return neighbor_squares


def flood_fill(chess_squares_list, square, region):
    if square is None or square not in chess_squares_list:
        return
    else:
        region.append(square)
        square_idx = chess_squares_list.index(square)
        chess_squares_list[square_idx] = None
        flood_fill(chess_squares_list, (square[0]-1, square[1]-1), region)
        flood_fill(chess_squares_list, (square[0]-1, square[1]), region)
        flood_fill(chess_squares_list, (square[0]-1, square[1]+1), region)
        flood_fill(chess_squares_list, (square[0], square[1]-1), region)
        flood_fill(chess_squares_list, (square[0], square[1]+1), region)
        flood_fill(chess_squares_list, (square[0]+1, square[1]-1), region)
        flood_fill(chess_squares_list, (square[0]+1, square[1]), region)
        flood_fill(chess_squares_list, (square[0]+1, square[1]+1), region)


def cool():
    board = chess.Board().empty()
    
    num_files = 8
    num_ranks = 8

    chess_squares_matrix = [[(file, rank) for file in range(0, num_files, 1)] for rank in range(num_ranks-1, -1, -1)]
    chess_squares_list = list(chain.from_iterable(chess_squares_matrix))

    # KING
    king_square = random.choice(chess_squares_list)
    board.set_piece_at(chess.square(king_square[0], king_square[1]), chess.Piece(chess.KING, chess.WHITE))
    print(king_square, "\n", board, "\n")

    # NEIGHBORS
    neighbor_squares = get_neighbor_squares(king_square)
    piece_square = random.choice(neighbor_squares)
    board.set_piece_at(chess.square(piece_square[0], piece_square[1]), chess.Piece(chess.ROOK, chess.WHITE))
    print(piece_square, "\n", board, "\n")

    # REGIONS
    for square_idx, square in enumerate(chess_squares_list):
        if board.piece_at(chess.square(square[0], square[1])) is not None:
            chess_squares_list[square_idx] = None
            # print(board, "\n")
        else:
            board.set_piece_at(chess.square(square[0], square[1]), chess.Piece(chess.KING, chess.BLACK))
            if not board.is_valid():
                chess_squares_list[square_idx] = None
                # print(board, "\n")
            board.remove_piece_at(chess.square(square[0], square[1]))
    # print(chess_squares_list)
    
    for i in range(8):
        print(chess_squares_list[i*8 : i*8+8])
    print('\n')

    regions = []
    for square in chess_squares_list:
        if square is not None:
            region = []
            flood_fill(chess_squares_list, square, region)
            if region:
                regions.append(region)
                for i in range(8):
                    print(chess_squares_list[i*8 : i*8+8])
                print('\n')



def main():
    cool()

if __name__ == "__main__":
    main()
