import argparse
from conf import Conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--conf", required=True, help="path of the configuration file")
    args = vars(ap.parse_args())

    config = args["conf"]

    fr_mo = Conf(config)

    for fram, mod in fr_mo["frameworks"]:
        #aqui hay que crear un .sh para cada modelo que haya y meterle la llamada al train model con las variables necesarias

        pass




if __name__ == "__main__":
    main()