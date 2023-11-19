import struct
import csv

#逆にして文字列として返す
def reverse_hex(hex_value):
    #bytesをstrに変換
    input_str = hex_value.decode('utf-8')
    input_str = input_str.replace("\n", "")
    #AABBCCDDをDDCCBBAAにする
    hex_value = input_str[6:] + input_str[4:6] + input_str[2:4] + input_str[0:2]
    if hex_value != '':
        return int(hex_value, 16)
    else:
        return -1

def recieve_data(s):
    try:
        data = s.recv(8)
        return data
    except Exception as e:
        print(f'エラーが発生しました: {e}')
        exit()