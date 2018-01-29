import sys
sys.path.append("/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/")
import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse import vstack as vstack
import chess, chess.pgn
import time
import multiprocessing as mp
import os
import logging
import pandas as pd
from datetime import datetime
import pickle
import gzip



# constants
BOARD_LENGTH = 768 #chess board is 8 x 8 and 12 different pieces
PIECE_OFFSETS = {'P': 0, 'N': 64, 'B': 128, 'R': 192, 'Q': 256, 'K': 320,
                 'p': 384, 'n': 448, 'b': 512, 'r': 576, 'q': 640, 'k': 704}
RESULTS = {'1-0': 1,'1/2-1/2': 0,'0-1': -1}



def parseFile(pgnFileName, inDir, outDir):
    '''Parses the games in the pgn file and saves the parsed games to disk

    Args:
        pgnFileName - str: name of the pgn file to parse
        inDir - str: path to the directory with the pgn files
        outDir - str: path to the directory where to save the games

    '''
    
    print('process {} starting'.format(mp.current_process().name))

    startTime = time.time()
          
    handle = open(os.path.join(inDir, pgnFileName))
    
    # dataframe to store statistics about the games in the pgn file
    # largest file has at most 1M games
    gamesStats = pd.DataFrame(data = 0,
                              index = range(1000000),
                              columns = ['Moves','Clock','RatingDiff','Result'])
    
    # sparse matrix to hold sequence of boards from all games
    gamesMat = csr_matrix((0, BOARD_LENGTH), dtype = np.int8)
    
    #numpy array to hold result
    gamesRes = np.zeros((0,1), dtype = np.int8)
    
    # read and parse games
    print('process {} reading games'.format(mp.current_process().name))
          
    totGames = 0

    while True:
        try:
            game = chess.pgn.read_game(handle)
        except ValueError:
            #try next game - return to the start of the loop
            continue
        else:
            break

    while game != None:
        totGames = totGames + 1
        
        # get sparse matrix, result and stats for game
        gameMat, gameRes, gameStats = parseGame(game)
 
        # save
        if gameMat != None:
            gamesMat = vstack([gamesMat, gameMat], format = 'csr')
            gamesRes = np.vstack([gamesRes,gameRes])
            gamesStats.loc[totGames-1,:] = gameStats

        if totGames % 10000 == 0:
            print('process {0} read {1} games'.format(
                mp.current_process().name, totGames))

        while True:
            try:
                game = chess.pgn.read_game(handle)
            except ValueError:
                #try next game - return to the start of the loop
                continue
            else:
                break

    handle.close()

    print('process {} done reading'.format(mp.current_process().name))

    print('process {} saving'.format(mp.current_process().name))

    # delete excess rows from data frame
    gamesStats = gamesStats.truncate(after = totGames-1, copy = False)
    
    # save processed file
    name_ = pgnFileName.rstrip('.pgn')
    outMat = os.path.join(outDir,name_) + '_mat.pickle.gz'
    outStats = os.path.join(outDir,name_) + '_stats.pickle'
    
    with gzip.open(outMat, 'wb') as f:
        f.write(pickle.dumps([gamesMat,gamesRes], pickle.HIGHEST_PROTOCOL))
    
    gamesStats.to_pickle(outStats)

    endTime = time.time()

    print('process {} ending'.format(mp.current_process().name))
    print('process {0} run time {1}'.format(mp.current_process().name, endTime - startTime))
          
    return()

def parseGame(game):
    '''parseGame(game object) -> (sparse matrix, np.array, list)
        
        game object = object of Game class from chess,
        sparse matrix = scipy csr matrix encoding the sequence of board positions,
        np.array = numpy array of result of game
        list = length 4, statistics about the game
        
    This function loops through the moves in a game and
    converts the sequence of board positions to a sparse matrix.
    In addition returns statistics about the game.
    Returns (None, None, None) if game has no half moves.
    '''
    
    board = game.board()
    
    #default values
    boardMatrix = None
    resultMatrix = None
    stats = [0]*4
    boardList = []
    
    mainLine = game.main_line()
    for i, halfMove in enumerate(mainLine): #mainLine is a generator object
        
        if i == 0:
            # get stats about game from headers
            try:
                h = game.headers
                stats[0] = int(h['PlyCount'])
                clock = h['WhiteClock']
                if clock.count(':') == 2:
                    stats[1] = int(datetime.strptime(clock,'%H:%M:%S.%f').minute)
                else:
                    stats[1] = int(datetime.strptime(clock,'%M:%S.%f').second/60)
                    
                stats[2] = int(h['WhiteElo']) - int(h['BlackElo'])
                stats[3] = RESULTS[h['Result']]
            except (KeyError, TypeError, ValueError):
                break #break out of for loop and skip game
                #stats = [np.nan]*4
        
        # convert sequence of board positions into 2-d binary array
        board.push(halfMove)
        #combine board vector with result
        boardList.append(convertBoardToVec(board))
    
    if boardList != []:
        boardMatrix = csr_matrix(np.stack(boardList, axis = 0), dtype=np.int8) #sparse array
        resultMatrix = np.full((boardMatrix.shape[0],1), stats[3], dtype=np.int8)
        #M = np.stack(boardList, axis = 0) #dense array
    
    return(boardMatrix, resultMatrix, stats)

def convertBoardToVec(board):
    '''convertBoardToVec(board object) -> array
        
        board object = object of Board Class from chess,
        array = 1d np array of length BOARD_LENGTH
        
    This function loops converts a board to its corresponding vector representation
    '''
    
    v = np.zeros(BOARD_LENGTH, dtype = np.int8)

    pieces = board.piece_map()
    for sq in pieces:
        piece = pieces[sq]
        ind = PIECE_OFFSETS[piece.symbol()] + sq
        v[ind] = 1
        
    return(v)

if __name__ == '__main__':
    inDataDir = sys.argv[1]
    outDataDir = sys.argv[2]

    mp.log_to_stderr()
    logger = mp.get_logger()
    logger.setLevel(logging.INFO)

    inDataFiles = os.listdir(inDataDir)

    startTime = time.time()

    workers = []

    i = 0
    for inFile in inDataFiles:
        if inFile.endswith('.pgn'):
            i = i + 1
            worker = mp.Process(name = 'file-' + str(i),
                                target = parseFile,
                                args = (inFile,inDataDir,outDataDir))
            workers.append(worker)
            worker.start()
    
    for worker in workers:
        worker.join()

    endTime = time.time()
    print('Total running time: {}'.format(endTime-startTime))
