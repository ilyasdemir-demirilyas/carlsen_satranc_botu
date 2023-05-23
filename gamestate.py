'''
burası kodlamanın ilk kısmı .burada GameState fonksyonunun görevi anlatılacaktır .
GameState fonksyonunu koşul ve durumlara göre 0 ve 1 lerden oluşan taş durum kodlamasını yapar yani 
eğer rok yapabiliyorsa 1 gibi bir kodlama yapabilirken yoksa 0 yapabilir .bu gibi durumlar sonucu toplam 14 adet binaryden oyun durum ortaya çıkar .

'''
import chess
import numpy as np

class GameState(object):
    def __init__(self, board=None):# burdaki ana fonksyonda satranç tahtası alınırı ,yoksa tahta oluşturularak işlemlere devam edilir . 
        if board==None:
            self.board = chess.Board()
        else:
            self.board = board

    def value(self):
        return 0


    def successors(self): #burada kurallara uygun yapılabilecek hareketlerin listesi döndürülür .
        return list(self.board.legal_moves)

    def bit_encode(self):
        #bir tahta durumunu 1'ler veya 0'lar dizisi olan 774 "bit dizisi" olarak kodlar.
        #6 taş * 64 kare * 2 renk = 768 + rok/harekete geçmek/geçerken alma için 6 bit = 774 bit

        bitstring = []
        for num in range(64):

            if self.board.piece_at(num) == None:
                empty = [0,0,0,0,0,0,0,0,0,0,0,0]
                bitstring = bitstring + empty
                continue
            bit_map = {"P": [1,0,0,0,0,0,0,0,0,0,0,0], "N": [0,1,0,0,0,0,0,0,0,0,0,0], "B": [0,0,1,0,0,0,0,0,0,0,0,0], "R": [0,0,0,1,0,0,0,0,0,0,0,0], "Q": [0,0,0,0,1,0,0,0,0,0,0,0], "K": [0,0,0,0,0,1,0,0,0,0,0,0], "p": [0,0,0,0,0,0,1,0,0,0,0,0], "n":[0,0,0,0,0,0,0,1,0,0,0,0], "b":[0,0,0,0,0,0,0,0,1,0,0,0], "r":[0,0,0,0,0,0,0,0,0,1,0,0], "q":[0,0,0,0,0,0,0,0,0,0,1,0], "k": [0,0,0,0,0,0,0,0,0,0,0,1]}
            temp = self.board.piece_at(num)
            type_of_piece = temp.symbol()
            piece = bit_map[type_of_piece]
            bitstring = bitstring + piece
            continue

        

        if self.board.has_queenside_castling_rights(chess.WHITE): #beyaz vezir tarafından rok yapma durumunu ifade eder .
            bitstring.append(1) 
        else:
            bitstring.append(0)

        if self.board.has_kingside_castling_rights(chess.WHITE): #beyaz şah tarafından rok yapma durumunu ifade eder .
            bitstring.append(1)
        else:
            bitstring.append(0)
        if self.board.has_queenside_castling_rights(chess.BLACK): #siyah vezir tarafından rok yapma durumunu ifade eder .
            bitstring.append(1)
        else:
            bitstring.append(0)
        if self.board.has_kingside_castling_rights(chess.BLACK):#siyah şah tarafından rok yapma durumunu ifade eder .
            bitstring.append(1)
        else:
            bitstring.append(0)

        if self.board.has_legal_en_passant():# geçerken alma durumu .
            bitstring.append(1)
        else:
            bitstring.append(0)
        if self.board.turn == chess.WHITE: # oynayan kişi 
            bitstring.append(1)
        else:
            bitstring.append(0)
        return bitstring