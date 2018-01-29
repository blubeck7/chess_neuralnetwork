import numpy as np
import chess
from random import shuffle
from random import seed
import signal

class TimeoutException(Exception):
    pass

class ComputerPlayer():

    RESULTS = np.array([[1,0,-1]], dtype = np.int8)
    RESULTSDICT = {'1-0': 1,'1/2-1/2': 0,'0-1': -1}
    BOARDLENGTH = 768
    PIECEOFFSETS = {'P': 0, 'N': 64, 'B': 128, 'R': 192, 'Q': 256, 'K': 320,
                    'p': 384, 'n': 448, 'b': 512, 'r': 576, 'q': 640, 'k': 704}

    def __init__(self, hiddenWeights, hiddenBiases,
                 outputWeights, outputBiases, color):
        
        self.numInputNodes = hiddenWeights.shape[0]
        self.numHiddenNodes = hiddenWeights.shape[1]
        self.numOutputNodes = outputWeights.shape[1]

        self.hiddenWeights = hiddenWeights
        self.hiddenBiases = hiddenBiases
        self.outputWeights = outputWeights
        self.outputBiases = outputBiases

        # parameters to vary difficulty level
        # maxtime = max number of seconds to search for move
        # depth = number of half moves deep to search
        # maxmoves = max number of moves to consider each turn
        #            zero means consider all moves
        # ordermth = method to order moves
        #            'rand' = random
        #            'score0' = by their no-depth scores
        # threshold = minimum no-depth score for move to be considered for
        #             full search
        self.comp = {}
        self.comp['maxtime'] = 5
        self.comp['depth'] = 4
        self.comp['maxmoves'] = 0
        self.comp['ordermth'] = 'rand'
        #self.comp['threshold'] = 0
        
        self.player = {}
        self.player['maxmoves'] = 0
        self.player['ordermth'] = 'rand'
        #self.player['threshold'] = 0

        #self.debug = True
        self.randState = 1

        self.color = 1
        if color == 'b':
            self.color = -1
        
        

    @classmethod
    def fromFile(cls, filePath, color):
        '''
        filePath - str: full path of the file that contains the neural network
            parameters
        color - str: w = white, b = black

        Loads the neural network parameters from the file and returns
        a ComputerPlayer object
        '''

        with np.load(filePath) as params:
            hiddenWeights = params['hiddenWeights']
            hiddenBiases = params['hiddenBiases']
            outputWeights = params['outputWeights']
            outputBiases = params['outputBiases']
            
        return cls(hiddenWeights, hiddenBiases, outputWeights, outputBiases,
                   color)

    def setDiff(self, maxDepth, maxTime, randState, compMaxMoves, playMaxMoves):
        '''
        maxDepth - int: maximum number of half moves deep to search
        maxTime - int: maxmimum number of seconds to spend on the search. The
            search is terminated is it has not finished in the maximum time.
        randState - int: this parameter sets the random seed to randomize the
            moves. The main use of randomizing the moves is so that the
            computer will play in a varied way
        maxMoves - int: How many moves to consider on each turn. 0 for all
            
        Sets the parameters that control the computer's difficulty
        '''

        self.comp['maxtime'] = maxTime
        self.comp['depth'] = maxDepth
        self.comp['maxmoves'] = compMaxMoves
        self.player['maxmoves'] = playMaxMoves
        self.randState = randState

    def predict(self, boardVec):
        '''
        boardVec - array: array of the encoded board position
    
        returns array of probability of win, draw or loss
        '''
        
        hiddenIn = boardVec.dot(self.hiddenWeights) + self.hiddenBiases
        hiddenOut = self._relu(hiddenIn)
        outputIn = hiddenOut.dot(self.outputWeights) + self.outputBiases
        outputOut = self._softmax(outputIn)
        
        return(outputOut)

    def _relu(self, X):
        '''
        X - array
        
        returns elementwise max of X and zero
        '''
        
        return(np.maximum(X,0))
    
    def _softmax(self, X):
        '''
        X - array
        returns array of probabilities
        '''
        shiftX = X - np.amax(X, axis = 1, keepdims = True)
        exps = np.exp(shiftX)
        sums = np.sum(exps, axis = 1, keepdims = True)
        
        return(exps / sums)

    def score(self, board, gameOver = False):
        '''
        board - chess.Board object

        Returns the 100 * the expected value for white. 1 = white win, 0 = draw,
            -1 = white loss
        '''

        if gameOver == True:
            score = 100 * self.RESULTSDICT[board.result()]
        else:
            boardVec = self.convertBoardToVec(board)
            probs = self.predict(boardVec)
            score = 100 * np.sum(probs * self.RESULTS)

        score = self.color * score
        return(score)
        

    def convertBoardToVec(self, board):
        '''
        board - chess.Board object
        
        Returns a 1 x 768 array that represents the board
        '''
    
        boardVec = np.zeros(self.BOARDLENGTH, dtype = np.int8)

        pieces = board.piece_map()
        for square in pieces:
            piece = pieces[square]
            ind = self.PIECEOFFSETS[piece.symbol()] + square
            boardVec[ind] = 1

        return(boardVec)

    def treePrint(self, depth, alpha, beta, isMax,
                  moves, score = None, init = False):
        if init == True:
            for i in range(self.comp['depth']):
                print('Depth ' + str(self.comp['depth'] - i) + '\t', end = '')
            print('Depth ' + str(0))

        spacing = '\t' * (self.comp['depth'] - depth)
        node = spacing + str(moves)
        node = node + ', a = ' + str(alpha) + ', b = ' + str(beta)
        node = node + ', ' + str(isMax)

        if score != None:
            node = node + ', s = ' + str(score)
            
        print(node)

    def printInfo(self, board, depth, alpha, beta, isMax, moves):
        print()
        print(board)
        print('Current Depth = ' + str(depth))
        print('Current Alpha = ' + str(alpha))
        print('Current Beta = ' + str(beta))
        print('Maximizer ' + str(isMax))
        if moves != []:
            print('Moves ' + str(moves))

    def orderMoves(self, board, moves, isMax, ordermth = None):
        '''
        board - chess.Board object: current board position
        moves - list: list of legal moves to consider
        isMax - bool: True = comp, False = player
        ordermth - str: can override global ordering parameter

        returns list of moves according to the order setting
        '''

        board2 = board.copy()
        orderedMoves = []
        
        if isMax == True:
            if ordermth == None:
                ordermth = self.comp['ordermth']
                
            if ordermth == 'score0':
                for move in moves:
                    board2.push(move)
                    orderedMoves.append((move, self.score(board2)))
                    board2.pop()
                orderedMoves.sort(key = lambda p: p[1], reverse = True)
                orderedMoves = [move[0] for move in orderedMoves]
            elif ordermth == 'rand':
                seed(self.randState)
                orderedMoves = moves.copy()
                shuffle(orderedMoves)
        else:
            if ordermth == None:
                ordermth = self.player['ordermth']
                
            if ordermth == 'score0':
                for move in moves:
                    board2.push(move)
                    orderedMoves.append((move, self.score(board2)))
                    board2.pop()
                orderedMoves.sort(key = lambda p: p[1], reverse = False)
                orderedMoves = [move[0] for move in orderedMoves]
            elif ordermth == 'rand':
                seed(self.randState)
                orderedMoves = moves.copy()
                shuffle(orderedMoves)
            
                               
        return(orderedMoves)
                  
    def turnMoves(self, board, isMax, ordermth = None):
        '''

        returns a list of moves to consider for the turn
        '''

        moves = [move for move in board.legal_moves]
        moves = self.orderMoves(board, moves, isMax, ordermth)
        if isMax == True:
            if self.comp['maxmoves'] > 0:
                moves = moves[:self.comp['maxmoves']]
        else:
            if self.player['maxmoves'] > 0:
                moves = moves[:self.player['maxmoves']]            

        return(moves)

        
    def getMove(self, board):
        '''
        board - chess.Board object

        returns the computer's move. The minimax algorithm with alpha-beta
        pruning is used to determine the computer's best move. If max time
        elapses, the search is interrupted and the current best move is
        returned
        '''

        alpha = -100 # best value maximizer can currently guarantee
        beta = 100 # best value minimizer can currently guarantee
        depth = self.comp['depth']
        isMax = True
        moveList = []
        bestMoveList = []

        #self.printInfo(board, depth, alpha, beta, isMax, moveList)
        #self.treePrint(depth, alpha, beta, isMax, moveList, init = True)

        # start move timer
        signal.signal(signal.SIGALRM, self.timeoutHandler)
        signal.alarm(self.comp['maxtime'])
                     
        try:
            # get moves
            moves = self.turnMoves(board, isMax)
            bestMove = moves[0]

            # evaluate moves        
            for move in moves:
                moveList.append(board.san(move))
                board.push(move)
                score, b = self.evalMove(board, depth - 1, alpha, beta,
                                         not isMax, moveList)
                board.pop()
                moveList.pop()
                #self.printInfo(board, depth, alpha, beta, isMax, moveList)
                #self.treePrint(depth, alpha, beta, isMax, moveList)
                
                if score > alpha:
                    bestMove = move
                    alpha = score
                    bestMoveList = b

            signal.alarm(0) # cancel scheduled alarm
            return(bestMove, alpha, bestMoveList)

        except TimeoutException:
            signal.alarm(0) # cancel scheduled alarm
            print('Timed out')
            return(bestMove, alpha, bestMoveList)

    def evalMove(self, board, depth, alpha, beta, isMax, moveList):
        '''
        board - chess.Board object: node to be evaluated
        depth - int: current level of the tree
        alpha - float: maximum lower bound of possible moves
        beta - float: minimum upper bound of possible moves
        isMax - bool: maximizing or minimizing player

        Returns the value of a move and its seq.
        Implements the minimax algorithm with alpha-beta pruning
        to determine the value of a particular move.
        '''
        
        if depth != 0:
            #self.printInfo(board, depth, alpha, beta, isMax, moveList)
            #self.treePrint(depth, alpha, beta, isMax, moveList)
            pass
        
        # check if terminal node or min depth
        if board.is_game_over() == True or depth == 0:
            score = self.score(board)
            #self.treePrint(depth, alpha, beta, isMax, moveList, score)
            return(score, moveList.copy())

        # find best move for player whose move it is
        if isMax == True:
            bestMoveList = []
            moves = self.turnMoves(board, isMax)
            
            for move in moves:
                moveList.append(board.san(move))
                board.push(move)
                score, b = self.evalMove(board, depth - 1, alpha, beta,
                                         False, moveList)

                board.pop()
                moveList.pop()
                #self.printInfo(board, depth, alpha, beta, isMax, moveList)
                #self.treePrint(depth, alpha, beta, isMax, moveList)
                
                if score > alpha:
                    alpha = score
                    bestMoveList = b
                    
                if beta <= alpha:
                    #print('Pruned')
                    break
                
            return(alpha, bestMoveList)
            
        else:
            bestMoveList = []
            moves = self.turnMoves(board, isMax)

            for move in moves:
                moveList.append(board.san(move))
                board.push(move)
                score, b = self.evalMove(board, depth - 1, alpha, beta,
                                         True, moveList)
                
                board.pop()
                moveList.pop()
                #self.printInfo(board, depth, alpha, beta, isMax, moveList)
                #self.treePrint(depth, alpha, beta, isMax, moveList)
                
                if score < beta:
                    beta = score
                    bestMoveList = b
                    
                if beta <= alpha:
                    #print('Pruned')
                    break

            return(beta, bestMoveList)

    def timeoutHandler(self, sigNum, frame):
        raise TimeoutException
        
if __name__ == '__main__':
    computer = ComputerPlayer.fromFile('../model/model_full_weights.npz', 'w')
 #   board = chess.Board()
#    m = computer.getMove(board)
#    print("Computer's best move: " + str(m))
