"""
ConvLSTM2D unroll theo thoi gian (khong co op Loop) de export ONNX -> QNN.

Dung:
    from unrolled_convlstm import convert_model
    new_model = convert_model(old_model)      # old_model: model cu DA TRAIN (load bang load_model)
    # new_model co cung trong so, chi khac: ConvLSTM2D -> UnrolledConvLSTM2D
"""
import numpy as np
import tensorflow as tf


class UnrolledConvLSTM2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, padding="valid", return_sequences=False,
                 activation="tanh", recurrent_activation="sigmoid", use_bias=True, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = tuple(kernel_size)
        if any(k % 2 == 0 for k in self.kernel_size):
            raise ValueError("Chi ho tro kernel le.")
        self.padding = padding.upper()
        self.return_sequences = return_sequences
        self.activation = tf.keras.activations.get(activation)
        self.recurrent_activation = tf.keras.activations.get(recurrent_activation)
        self.use_bias = use_bias

    def build(self, input_shape):
        c = int(input_shape[-1])
        kh, kw = self.kernel_size
        self.kernel = self.add_weight(name="kernel", shape=(kh, kw, c, 4 * self.filters),
                                      initializer="glorot_uniform")
        self.recurrent_kernel = self.add_weight(name="recurrent_kernel",
                                                shape=(kh, kw, self.filters, 4 * self.filters),
                                                initializer="orthogonal")
        if self.use_bias:
            self.bias = self.add_weight(name="bias", shape=(4 * self.filters,), initializer="zeros")

    def call(self, x, mask=None):
        T = x.shape[1]
        if T is None:
            raise ValueError("Chieu thoi gian phai co dinh de unroll.")
        h = c = None
        outs = []
        for t in range(T):
            z = tf.nn.conv2d(x[:, t], self.kernel, strides=1, padding=self.padding)
            if h is not None:
                z += tf.nn.conv2d(h, self.recurrent_kernel, strides=1, padding="SAME")
            if self.use_bias:
                z = tf.nn.bias_add(z, self.bias)
            i, f, g, o = tf.split(z, 4, axis=-1)
            i, f, o = (self.recurrent_activation(v) for v in (i, f, o))
            g = self.activation(g)
            c = i * g if c is None else f * c + i * g
            h = o * self.activation(c)
            outs.append(h)
        return tf.stack(outs, axis=1) if self.return_sequences else h

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            filters=self.filters,
            kernel_size=list(self.kernel_size),
            padding=self.padding.lower(),
            return_sequences=self.return_sequences,
            activation=tf.keras.activations.serialize(self.activation),
            recurrent_activation=tf.keras.activations.serialize(self.recurrent_activation),
            use_bias=self.use_bias,
        )
        return cfg


def _swap(layer):
    if isinstance(layer, tf.keras.layers.ConvLSTM2D):
        c = layer.get_config()
        ok = (tuple(c["strides"]) == (1, 1) and tuple(c["dilation_rate"]) == (1, 1)
              and not c["go_backwards"] and c["data_format"] == "channels_last"
              and c.get("dropout", 0) == 0 and c.get("recurrent_dropout", 0) == 0)
        if not ok:
            raise ValueError(f"{layer.name}: cau hinh khac mac dinh, class unroll chua ho tro: {c}")
        return UnrolledConvLSTM2D(c["filters"], c["kernel_size"], c["padding"],
                                  c["return_sequences"], c["activation"],
                                  c["recurrent_activation"], c["use_bias"], name=layer.name)
    return layer.__class__.from_config(layer.get_config())


def convert_model(old_model, atol=1e-5, n_check=2):
    """Thay ConvLSTM2D bang ban unroll, copy trong so theo ten layer, roi so output."""
    new_model = tf.keras.models.clone_model(old_model, clone_function=_swap)
    for layer in new_model.layers:
        if layer.weights:
            layer.set_weights(old_model.get_layer(layer.name).get_weights())

    rng = np.random.default_rng(0)
    xs = [rng.normal(size=(n_check, *i.shape[1:])).astype("float32") for i in old_model.inputs]
    diff = float(np.abs(old_model(xs).numpy() - new_model(xs).numpy()).max())
    print(f"max|diff| model cu vs moi = {diff:.2e}")
    if diff > atol:
        raise RuntimeError("Output lech - kiem tra cau hinh ConvLSTM2D (recurrent_activation, strides...).")
    return new_model
