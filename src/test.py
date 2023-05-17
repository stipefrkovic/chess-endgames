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


class ChessSquare():
    num_files = 8
    num_ranks = 8

    def __init__(self, file, rank):
        self.file = file
        self.rank = rank

    def __repr__(self):
        return f"({self.file}, {self.rank})"
    
    def __eq__(self, other): 
        if not isinstance(other, ChessSquare):
            return NotImplemented
        return self.file == other.file and self.rank == other.rank

    
    def create_square(self):
        return chess.square(self.file, self.rank)
    
    def legal_neighbors(self):
        neighbor_squares = []
        # Check bottom-left neighbor
        if self.rank > 0 and self.file > 0:
            neighbor_squares.append(ChessSquare(self.file-1, self.rank-1))
        # Check bottom neighbor
        if self.rank > 0:
            neighbor_squares.append(ChessSquare(self.file, self.rank-1))
        # Check bottom-right neighbor
        if self.rank > 0 and self.file < self.num_files - 1:
            neighbor_squares.append(ChessSquare(self.file+1, self.rank-1))
        # Check left neighbor
        if self.file > 0:
            neighbor_squares.append(ChessSquare(self.file-1, self.rank))
        # Check right neighbor
        if self.file < self.num_files - 1:
            neighbor_squares.append(ChessSquare(self.file+1, self.rank))
        # Check top-left neighbor
        if self.rank < self.num_ranks - 1 and self.file > 0:
            neighbor_squares.append(ChessSquare(self.file-1, self.rank+1))
        # Check top neighbor
        if self.rank < self.num_ranks - 1:
            neighbor_squares.append(ChessSquare(self.file, self.rank+1))
        # Check top-right neighbor
        if self.rank < self.num_ranks - 1 and self.file < self.num_files - 1:
            neighbor_squares.append(ChessSquare(self.file+1, self.rank+1))
        return neighbor_squares

class ChessBoard():
    def __init__(self, squares):
        self.squares = squares

    def random_square(self):
        square = random.choice(self.squares)
        return square
    
    def remove_illegal_squares(self, board):
        for square_idx, square in enumerate(self.squares):
            if board.piece_at(square.create_square()) is not None:
                self.squares[square_idx] = None
            else:
                board.set_piece_at(square.create_square(), chess.Piece(chess.KING, chess.BLACK))
                if not board.is_valid():
                    self.squares[square_idx] = None
                board.remove_piece_at(square.create_square())
        # for i in range(8):
        #     print(self.squares[i*8 : i*8+8])
        # print('\n')


    def flood_fill(self, square, region):
        if square is None or square not in self.squares:
            return
        else:
            region.append(square)
            square_idx = self.squares.index(square)
            self.squares[square_idx] = None
            neighbors = square.legal_neighbors()
            for neighbor in neighbors:
                self.flood_fill(neighbor, region)

    def find_smallest_region(self):
        regions = []
        for square in self.squares:
            if square is not None:
                region = []
                self.flood_fill(square, region)
                if region:
                    regions.append(region)
                    # for i in range(8):
                    #     print(self.squares[i*8 : i*8+8])
                    # print('\n')
        # for region in regions:
        #     print(region)
        return min(regions, key=len)



def cool():
    chess_squares = ChessBoard([ChessSquare(file, rank) for rank in range(ChessSquare.num_ranks-1, -1, -1) for file in range(0, ChessSquare.num_files, 1)])
    board = chess.Board().empty()

    # KING
    king_square = chess_squares.random_square()
    board.set_piece_at(king_square.create_square(), chess.Piece(chess.KING, chess.WHITE))

    # NEIGHBORS
    neighbor_squares = ChessBoard(king_square.legal_neighbors())
    piece_square = neighbor_squares.random_square()
    board.set_piece_at(piece_square.create_square(), chess.Piece(chess.ROOK, chess.WHITE))

    # REGION
    chess_squares.remove_illegal_squares(board)
    other_king_squares = ChessBoard(chess_squares.find_smallest_region())
    other_king_square = other_king_squares.random_square()
    board.set_piece_at(other_king_square.create_square(), chess.Piece(chess.KING, chess.BLACK))

    return board


def main():
    print(cool())

if __name__ == "__main__":
    main()
