'''
burada oyunlar önce oyuna ayrılır .
Oyuna ayrıldıktan sonra oyunlar denetlenir ve uygun hale getirilir .
uygun hale getirildikten sonra eğitim için numpy formatında bir dosyada kaydedilir  .
'''

import random
from gamestate import GameState
import os
import chess
import chess.pgn
import numpy as np
import csv

def parse():#burada oyun dosyaları tek tek oyunlara çevrilir .
    games = []
    values = []
    for pgnfile in os.listdir("games"):
        pgn = open(os.path.join("games", pgnfile))
        count = 0
        while True:
            try:
                subpgn = chess.pgn.read_game(pgn)
                
            except:
                break

            value_assign_dict = {'1-0': 1, '0-1': -1, '1/2-1/2':0}
            #Oyun sonuçlarını değerlere atar; beyazın galibiyeti 1, siyahın galibiyeti -1, beraberlik 0'dır.
            try:
                game_value = value_assign_dict[subpgn.headers['Result']]
            except:
                break
            #daha verimli öğrenme için çekilişlere yer verilmeyecek; 0 oyun değerlerini çıkaracağız.

            tempboard = subpgn.board()
            #Subpgn'nin tüm hareketlerini tahtaya ekler.
            count += 1
            print("ayrıştırılan oyun numarası " + str(count))
            for move in subpgn.mainline_moves():

                tempboard.push(move)
                output = GameState(tempboard).bit_encode()
                print("GameState(tempboard).bit_encode() :",GameState(tempboard).bit_encode())
                games.append(output)
                values.append(game_value)


    games = np.array(games)
    values = np.array(values)
    print("parse :" ,games, values)
    return "aaa",games, values


def prepare_data():#burada veri seti istenilen formata uygunluğu kontrol edilerek hazırlanıyor .

    positions = []
    results = []
    x = 0

    for pgnfile in os.listdir("games"): # oyunlar burada kontrol edilir .
        pgn = open(os.path.join("games", pgnfile))
        value_assign_dict = {'1-0': [1,0], '0-1': [0,1], '1/2-1/2': [0,0]}

        while (True):
            try:
                subpgn = chess.pgn.read_game(pgn)
            except:
                break

            try:
                game_value = value_assign_dict[subpgn.headers['Result']]
            except:
                break

            if game_value[0] == 0 and game_value[1] == 0:
                continue

            upper_bound = subpgn.end().ply()

            if upper_bound <= 5:
                continue

            tempboard = subpgn.board()
            mainline_moves = subpgn.mainline_moves()

            for j in range(10):
                tempboard.reset()
                move_number = random.randint(5, upper_bound)
                iterator = iter(mainline_moves)
                for i in range(move_number):
                    tempboard.push(next(iterator))
                trian_game_value = game_value[0]
                positions.append(GameState(tempboard).bit_encode())
                results.append(trian_game_value)
                x+=1

    positions = np.array(positions)
    results = np.array(results)
    np.save('./data/positions.npy', positions)
    np.save('./data/results.npy', results)
    print("oyun sayısı :",x)