from parameter import *
from wordvec import *
import data
import rnn_model
import train
import cifar100vgg as cfvg


# 记录在运行程序时后面的字符串，默认是default
def defineArgs():
    parser = argparse.ArgumentParser(description="Chinese_poem_generator.")
    parser.add_argument("-m", "--mode", help="select mode by 'train' or test or head",
                        choices=["train", "test", "head"], default="test")
    return parser.parse_args()


if __name__ == "__main__":
    args = defineArgs()
    # 初始化数据的对象
    traindata = data.poems_data(isEvaluate=False)
    # 进行训练或者生成
    if args.mode == "train":
        train.train(traindata)
    else:
        print("请输入图片的路径:")
        path = input()
        label = cfvg.pic_to_label(path, show_pictures=False)
        print(label)
        poems = train.label_poem(label)
