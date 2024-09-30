import os

class DisplayOutputs(keras.callbacks.Callback):
    def __init__(
        self, batch, idx_to_token, target_start_token_idx=27, target_end_token_idx=28
    ):
        """Displays a batch of outputs after every epoch

        Args:
            batch: A test batch containing the keys "source" and "target"
            idx_to_token: A List containing the vocabulary tokens corresponding to their indices
            target_start_token_idx: A start token index in the target vocabulary
            target_end_token_idx: An end token index in the target vocabulary
        """
        self.batch = batch
        self.target_start_token_idx = target_start_token_idx
        self.target_end_token_idx = target_end_token_idx
        self.idx_to_char = idx_to_token
        self.total_wer = 0.0
        self.num_batches = 0

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 != 0:
            return
        
        for batch in ds:
            source = batch["source"]
            target = batch["target"].numpy()
            bs = tf.shape(source)[0]
            preds = model.generate(source, self.target_start_token_idx)
            preds = preds.numpy()

            targets = []
            predictions = []
            for i in range(bs):
                target_text = "".join([self.idx_to_char[_] for _ in target[i, :]])
                prediction = ""
                for idx in preds[i, :]:
                    prediction += self.idx_to_char[idx]
                    if idx == self.target_end_token_idx:
                        break
                targets.append(target_text.replace("-", ""))
                predictions.append(prediction)
                print(f"target:     {target_text.replace('-','')}")
                print(f"prediction: {prediction}\n")
            
            # Save predictions to file
            with open("predictions.txt", "a") as f:
                f.write("Epoch {}\n".format(epoch))
                for target_text, prediction in zip(targets, predictions):
                    f.write("Target: {}\n".format(target_text))
                    f.write("Prediction: {}\n\n".format(prediction))
            
            batch_wer = wer(targets, predictions)
            self.total_wer += batch_wer
            self.num_batches += 1

    def on_train_end(self, logs=None):
        average_wer = self.total_wer / self.num_batches
        
        print("-" * 100)
        print(f"Average Word Error Rate: {average_wer:.2f}")
        print("-" * 100)
