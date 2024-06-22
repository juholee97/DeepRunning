import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
import tensorflow as tf


# (1) 데이터를 분류하기 위해 SLP 기반의 Binary Classifier, MLP 기반의 Binary Classifier 
# Keras_p1의 MLP
class MLP:
    def __init__(self, hidden_layer_conf, num_output_nodes):
        self.hidden_layer_conf = hidden_layer_conf
        self.num_output_nodes = num_output_nodes
        self.logic_op_model = None

    def build_model(self):
        input_layer = tf.keras.Input(shape=[2, ])
        hidden_layers = input_layer

        if self.hidden_layer_conf is not None:
            for num_hidden_nodes in self.hidden_layer_conf:
                hidden_layers = tf.keras.layers.Dense(units=num_hidden_nodes,
                                                    activation=tf.keras.activations.sigmoid,
                                                    use_bias=True)(hidden_layers)

        output = tf.keras.layers.Dense(units=self.num_output_nodes,
                                    activation=tf.keras.activations.sigmoid,
                                    use_bias=True)(hidden_layers)

        self.logic_op_model = tf.keras.Model(inputs=input_layer, outputs=output)

        # Learning rate = 0.1
        sgd = tf.keras.optimizers.SGD(learning_rate=0.1)
        self.logic_op_model.compile(optimizer=sgd, loss="mse")

    def fit(self, x, y, batch_size, epochs):
        self.logic_op_model.fit(x=x, y=y, batch_size=batch_size, epochs=epochs)

    def predict(self, x, batch_size):
        prediction = self.logic_op_model.predict(x=x, batch_size=batch_size)
        return prediction


def CircleClassify():
    # generating data
    n_samples = 400
    noise = 0.02
    factor = 0.5
    #### use x_train (Feature vectors), y_train (Class ground truths) as training set
    x_train, y_train = make_circles(n_samples=n_samples, noise=noise, factor=factor)
    #### use x_test (Feature vectors) as test set
    #### you do not use y_test for this assignment.
    x_test, y_test = make_circles(n_samples=n_samples, noise=noise, factor=factor)

    #### visualizing training data distribution

    batch_size= 1
    epochs = 300

    #(2) 각 Classifier를 Training data를 사용하여 학습하고,
    #Training
    SLP_model = MLP(hidden_layer_conf=None, num_output_nodes=1)
    SLP_model.build_model()
    SLP_model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs)

    MLP_model = MLP(hidden_layer_conf=[5, 5], num_output_nodes=1)
    MLP_model.build_model()
    MLP_model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs)

    # Training result
    SLP_predict = SLP_model.predict(x_test, batch_size=batch_size)
    SLP_result = []
    for i in SLP_predict:
        if i < 0.5:
            SLP_result.append(0)
        else:
            SLP_result.append(1)

    MLP_predict = MLP_model.predict(x_test, batch_size=batch_size)
    MLP_result = []
    for i in MLP_predict:
        if i < 0.5:
            MLP_result.append(0)
        else:
            MLP_result.append(1)

    # Print Result
    plt.subplot(121)
    plt.scatter(x_test[:, 0], x_test[:, 1], c=SLP_result, marker='.')
    plt.title("SLP_Training_Result")

    plt.subplot(122)
    plt.scatter(x_test[:, 0], x_test[:, 1], c=MLP_result, marker='.')
    plt.title("MLP_Training_Result")

    plt.savefig('Circle_Result/Result_5', dpi=300)
    plt.show()


if __name__ == '__main__':
    CircleClassify()
