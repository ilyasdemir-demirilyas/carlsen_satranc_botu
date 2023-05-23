'''
burada eğitim için hazırlanan verilerin gamestate sayesinde encode hale getirilerek eğitim için uygun formata getirilir ve 
prepare_data sayesinde numpy dosyasında tutulurlar .
'''

from gamestate import GameState
from training import prepare_data

print(prepare_data())
#134130 oyun .