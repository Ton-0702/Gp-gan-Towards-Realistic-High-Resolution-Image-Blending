import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda


def init_conv(array):
    xp = cuda.get_array_module(array)
    array[...] = xp.random.normal(loc=0.0, scale=0.02, size=array.shape)


def init_bn(array):
    xp = cuda.get_array_module(array)
    array[...] = xp.random.normal(loc=1.0, scale=0.02, size=array.shape)


class ReLU(chainer.Chain):
    def __init__(self):
        super(ReLU, self).__init__()

    def __call__(self, x):
        return F.relu(x)


class Tanh(chainer.Chain):
    def __init__(self):
        super(Tanh, self).__init__()

    def __call__(self, x):
        return F.tanh(x)


class LeakyReLU(chainer.Chain):
    def __init__(self):
        super(LeakyReLU, self).__init__()

    def __call__(self, x):
        return F.leaky_relu(x)


class DCGAN_G(chainer.ChainList):
    def __init__(self, isize, nc, ngf, conv_init=None, bn_init=None):
        cngf, tisize = ngf // 2, 4 # 64//2 =32, tisize= 4
        while tisize != isize: # nếu tisize != isize(64)
            cngf = cngf * 2 # lặp lai tới khi tisize = isize => sẽ có kết quả 512
            tisize = tisize * 2 # kết quả cuối cùng 64

        layers = []
        # input is Z, going into a convolution
        layers.append(L.Deconvolution2D(None, cngf, ksize=4, stride=1, pad=0, initialW=conv_init, nobias=True))
        layers.append(L.BatchNormalization(cngf, initial_gamma=bn_init))
        layers.append(ReLU())
        csize, cndf = 4, cngf
        while csize < isize // 2: # lặp nếu csize < isize (32)
            # Deconvolution2D là phục hồi kích thước ảnh
            layers.append(L.Deconvolution2D(None, cngf // 2, ksize=4, stride=2, pad=1, initialW=conv_init, nobias=True)) # vòng lặp đầu 512//2 =256, và lặp lại đến khi cize > isze
            layers.append(L.BatchNormalization(cngf // 2, initial_gamma=bn_init))
            layers.append(ReLU())
            cngf = cngf // 2
            csize = csize * 2
        layers.append(L.Deconvolution2D(None, nc, ksize=4, stride=2, pad=1, initialW=conv_init, nobias=True))
        layers.append(Tanh())

        super(DCGAN_G, self).__init__(*layers)

    def __call__(self, x):
        for i in range(len(self)):
            x = self[i](x)

        return x


class DCGAN_D(chainer.ChainList):
    def __init__(self, isize, ndf, nz=1, conv_init=None, bn_init=None):
        layers = []
        # ksize=4 (kernel (filter): 4x4), initialW khởi tạo weight Ban đầu, nobias= True (không thuật ngữ thiên vị)
        layers.append(L.Convolution2D(None, ndf, ksize=4, stride=2, pad=1, initialW=conv_init, nobias=True)) # sẽ ra kết quả tương tự csize, cndf
        #leakyRelu: những giá trị nhỏ hơn 0 sẽ vẫn được xét một độ dốc nhỏ cho các giá trị âm thay vì để giá trị là 0.
        layers.append(LeakyReLU())
        csize, cndf = isize / 2, ndf # csize sẽ từ 64 thành 32, cndf là 64 (dc xem là số channel) -- kết quả dc gán để làm điều kiện lặp
        while csize > 4: # nếu size lớn hơn 4 
            in_feat = cndf # đầu vào đặc trưng 64  
            out_feat = cndf * 2 # đầu ra đăc trưng của channel
            layers.append(L.Convolution2D(None, out_feat, ksize=4, stride=2, pad=1, initialW=conv_init, nobias=True))
            layers.append(L.BatchNormalization(out_feat, initial_gamma=bn_init))
            layers.append(LeakyReLU())

            cndf = cndf * 2 # cập nhật kết qua layers mới
            csize = csize / 2 # cập nhật kết qua layers mới
        # state size. K x 4 x 4
        layers.append(L.Convolution2D(None, nz, ksize=4, stride=1, pad=0, initialW=conv_init, nobias=True)) # kết quả của cuối là fatten thành vector dọc 1000

        super(DCGAN_D, self).__init__(*layers)

    def encode(self, x):
        for i in range(len(self)):
            x = self[i](x)

        return x

    def __call__(self, x):
        x = self.encode(x)
        x = F.sum(x, axis=0) / x.shape[0]
        return F.squeeze(x)


class EncoderDecoder(chainer.Chain):
    def __init__(self, nef, ngf, nc, nBottleneck, image_size=64, conv_init=None, bn_init=None):
        super(EncoderDecoder, self).__init__(
            encoder=DCGAN_D(image_size, nef, nBottleneck, conv_init, bn_init),
            bn=L.BatchNormalization(nBottleneck, initial_gamma=bn_init),
            decoder=DCGAN_G(image_size, nc, ngf, conv_init, bn_init)
        )

    def encode(self, x):
        h = self.encoder.encode(x)
        h = F.leaky_relu(self.bn(h))

        return h

    def decode(self, x):
        h = self.decoder(x)

        return h

    def __call__(self, x):
        h = self.encode(x)
        h = self.decode(h)
        return h
