import pandas as pd
import numpy as np
from transformers import BertTokenizer, TFAutoModel
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf


class CustomModel:

    def data_preparing():
        df = pd.read_csv('data/IMDB Dataset.csv')
        print(df.sample(5),df.shape)
        df_train=df[:45000]
        df_test=df[45000:]
        df_test.to_csv('data/test.csv',index=False)
        seq_len = 512
        num_samples = len(df_train)
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

        # Tokenize - this time returning NumPy tensors
        tokens = tokenizer(df_train['review'].tolist(), max_length=seq_len, truncation=True,
                            padding='max_length', add_special_tokens=True,
                            return_tensors='np')
        print(tokens.keys())
        with open('data/movie-xids.npy', 'wb') as f:
            np.save(f, tokens['input_ids'])
        with open('data/movie-xmask.npy', 'wb') as f:
            np.save(f, tokens['attention_mask'])
        
        labels = pd.get_dummies(df_train['sentiment']).values  # Automatically handles "positive"/"negative" as [1, 0] or [0, 1]

        print(labels.shape)
        with open('data/movie-labels.npy', 'wb') as f:
            np.save(f, labels)

    def train_model():
        with open('data/movie-xids.npy', 'rb') as f:
            Xids = np.load(f, allow_pickle=True)
        with open('data/movie-xmask.npy', 'rb') as f:
            Xmask = np.load(f, allow_pickle=True)
        with open('data/movie-labels.npy', 'rb') as f:
            labels = np.load(f, allow_pickle=True)

        # Ensure correct data types (convert to int32 for input tensors and float32 for labels)
        Xids = Xids.astype(np.int32)
        Xmask = Xmask.astype(np.int32)
        labels = labels.astype(np.float32)

        # Create TensorFlow dataset
        dataset = tf.data.Dataset.from_tensor_slices((Xids, Xmask, labels))
        print(dataset.take(1))

        def map_func(input_ids, masks, labels):
            return {'input_ids': input_ids, 'attention_mask': masks}, labels

        dataset = dataset.map(map_func)
        batch_size = 16
        dataset = dataset.shuffle(10000).batch(batch_size, drop_remainder=True)

        split = 0.9
        size = int((Xids.shape[0] / batch_size) * split)
        train_ds = dataset.take(size)
        val_ds = dataset.skip(size)

        tf.data.experimental.save(train_ds, 'train')
        tf.data.experimental.save(val_ds, 'val')

        # Specify correct element spec
        element_spec = ({
                'input_ids': tf.TensorSpec(shape=(None, 512), dtype=tf.int32),
                'attention_mask': tf.TensorSpec(shape=(None, 512), dtype=tf.int32)
            },
            tf.TensorSpec(shape=(None,2), dtype=tf.float32)
        )

        ds_train = tf.data.experimental.load('train', element_spec=element_spec)
        ds_val = tf.data.experimental.load('val', element_spec=element_spec)

        bert = TFAutoModel.from_pretrained('bert-base-cased')
        print(bert.summary())

        input_ids = tf.keras.layers.Input(shape=(512,), name='input_ids', dtype='int32')
        mask = tf.keras.layers.Input(shape=(512,), name='attention_mask', dtype='int32')

        # Access the transformer model within our BERT object using the bert attribute (e.g., bert.bert)
        embeddings = bert.bert(input_ids, attention_mask=mask)[1]  # Access final activations (already max-pooled)
        x = tf.keras.layers.Dense(1024, activation='relu')(embeddings)
        y = tf.keras.layers.Dense(2, activation='softmax', name='outputs')(x)

        model = tf.keras.Model(inputs=[input_ids, mask], outputs=y)

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, weight_decay=1e-6)
        loss = tf.keras.losses.CategoricalCrossentropy()
        acc = tf.keras.metrics.CategoricalAccuracy('accuracy')

        model.compile(optimizer=optimizer, loss=loss, metrics=[acc])

        history = model.fit(
            ds_train,
            validation_data=ds_val,
            epochs=3
        )

        model.save('sentiment_model')


# Uncomment to prepare data and train model
Model.data_preparing()
Model.train_model()

