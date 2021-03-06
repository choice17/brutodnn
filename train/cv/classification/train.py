from tensorflow.keras.optimizers import Adam
from tensorflow.python.framework.graph_util import convert_variables_to_constants
import tensorflow as tf
import tensorflow.keras.backend as K

class Train(object):
    def __init__(self):
        pass

    def buildTraining(self):
        pass

    def buildTrainKeras(self):
        self.model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

    def buildTrainTf(self):
        pass

    def fit(self):
        pass

    def eval(self):
        pass

    def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
        """
        Freezes the state of a session into a pruned computation graph.
    
        Creates a new computation graph where variable nodes are replaced by
        constants taking their current value in the session. The new graph will be
        pruned so subgraphs that are not necessary to compute the requested
        outputs are removed.
        @param session The TensorFlow session to be frozen.
        @param keep_var_names A list of variable names that should not be frozen,
                            or None to freeze all the variables in the graph.
        @param output_names Names of the relevant graph outputs.
        @param clear_devices Remove the device directives from the graph for better portability.
        @return The frozen graph definition.
        """
        graph = session.graph
        with graph.as_default():
            freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
            output_names = output_names or []
            output_names += [v.op.name for v in tf.global_variables()]
            input_graph_def = graph.as_graph_def()
            if clear_devices:
                for node in input_graph_def.node:
                    node.device = ""
            frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                        output_names, freeze_var_names)
            return frozen_graph

    def save(self, path_name, file_name):
        frozen_graph = Train.freeze_session(K.get_session(),
                              output_names=[out.op.name for out in self.model.outputs])
        tf.train.write_graph(frozen_graph, path_name, file_name, as_text=False)
        #self.saver_tf.save(self.sess, os.path.join(path_name,file_name), global_step=999)

