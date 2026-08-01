import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense

@tf.keras.utils.register_keras_serializable(package="custom")
class TiedLSTMLanguageModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, lstm_units, dropout_rate, sequence_length, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.sequence_length = sequence_length

        self.embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim,
                                    input_length=sequence_length, name="embedding")
        self.embedding_dropout = Dropout(dropout_rate, name="embedding_dropout")
        self.lstm_1 = LSTM(lstm_units, return_sequences=True, name="lstm_1")
        self.lstm_1_dropout = Dropout(dropout_rate, name="lstm_1_dropout")
        self.lstm_2 = LSTM(lstm_units, name="lstm_2")
        self.lstm_2_dropout = Dropout(dropout_rate, name="lstm_2_dropout")

        self.needs_projection = (lstm_units != embedding_dim)
        if self.needs_projection:
            self.projection = Dense(embedding_dim, activation=None, name="pre_output_projection")

        self.output_bias = self.add_weight(
            name="output_bias", shape=(vocab_size,), initializer="zeros", trainable=True
        )

    def call(self, inputs, training=False):
        x = self.embedding(inputs)
        x = self.embedding_dropout(x, training=training)
        x = self.lstm_1(x)
        x = self.lstm_1_dropout(x, training=training)
        x = self.lstm_2(x)
        x = self.lstm_2_dropout(x, training=training)
        if self.needs_projection:
            x = self.projection(x)
        logits = tf.matmul(x, self.embedding.embeddings, transpose_b=True) + self.output_bias
        return tf.nn.softmax(tf.cast(logits, tf.float32), axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "lstm_units": self.lstm_units,
            "dropout_rate": self.dropout_rate,
            "sequence_length": self.sequence_length,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)