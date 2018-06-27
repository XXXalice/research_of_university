from puckage import Puckage

def main():
    keyword = str(input('モデルを生成するキーワードを入力して下さい>> '))
    name = str(input('フォルダ名、モデル名に使用する英数名を入力して下さい>> '))
    p = Puckage(name, keyword)
    p.img_collect()
    p.makemodel()

if __name__ == '__main__':
    main()
