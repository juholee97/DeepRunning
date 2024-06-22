import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import PIL


def preprocessing_x(x_data):
    x_data = x_data / 255.0
    return x_data

def print_image(img_data):
    plt.figure()
    plt.imshow(img_data)
    plt.colorbar()
    plt.grid(False)
    plt.show()

class MLP:
    def __init__(self, hidden_layer_conf, num_output_nodes):
        self.hidden_layer_conf = hidden_layer_conf
        self.num_output_nodes = num_output_nodes
        self.logic_op_model = None

    def build_model(self):
        input_layer = tf.keras.Input(shape=[28, 28, ])
        flat_layer = tf.keras.layers.Flatten()(input_layer)
        flat_layer = preprocessing_x(flat_layer)
        hidden_layers = flat_layer

        if self.hidden_layer_conf is not None:
            for num_hidden_nodes in self.hidden_layer_conf:
                hidden_layers = tf.keras.layers.Dense(units=num_hidden_nodes,
                                                    activation=tf.keras.activations.relu,
                                                    use_bias=True)(hidden_layers)

        self.output = tf.keras.layers.Dense(units=self.num_output_nodes,
                                        activation=tf.keras.activations.softmax,
                                        use_bias=True)(hidden_layers)

        self.logic_op_model = tf.keras.Model(inputs=input_layer, outputs=self.output)

        opt_alg = tf.keras.optimizers.Adam(learning_rate=0.0003)
        categorical_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        self.logic_op_model.compile(optimizer=opt_alg, loss=categorical_loss)

    def fit(self, x, y, batch_size, epochs):
        self.logic_op_model.fit(x=x, y=y, batch_size=batch_size, epochs=epochs)

    def predict(self, x, batch_size):
        prediction = self.logic_op_model.predict(x=x, batch_size=batch_size)
        return prediction


def MNIST_Classify():
    mnist_train_data, mnist_test_data = tf.keras.datasets.mnist.load_data()
    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    #### use x_train (Images(Feature vectors)), y_train (Class ground truths) as training set
    x_train, y_train = mnist_train_data
    #### use x_test (Images(Feature vectors)), y_test (Class ground truths) as test set
    x_test, y_test = mnist_test_data

    batch_size = 1
    epochs = 10
    result = []


    for data in y_train:
        temp = [0 for i in range(10)]
        temp[data] = 1
        result.append(temp)
    y_train = np.array(result)

    M_model = MLP(hidden_layer_conf=[128, 128], num_output_nodes=10)
    M_model.build_model()
    M_model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs)

    predicted_labels = M_model.predict(x_test, batch_size)
    predicted_labels = tf.math.argmax(input=predicted_labels, axis=1)

    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_test[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[predicted_labels[i]])
    plt.savefig('MNIST_Result/MNIST_Result', dpi=300)


    
    #numpy를 이용하여 이미지를 배열객체로 변경
    input_data = []
    for i in range(10):
        input_img = PIL.Image.open("MNIST_input/"+str(i)+".png").convert("L")
        input_data.append(np.asarray(input_img))


    #색반전이 일어난 숫자 다시 색반전
    input_data = 255-np.array(input_data)

    my_predicted = M_model.predict(input_data, batch_size)
    my_predicted = tf.math.argmax(input=my_predicted, axis=1)

    plt.figure(figsize=(10, 10))
    for i in range(10):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(input_data[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[my_predicted[i]])
    plt.savefig('MNIST_Result/My_Num_Result', dpi=300)
    plt.show()


if __name__ == '__main__':
    MNIST_Classify()
