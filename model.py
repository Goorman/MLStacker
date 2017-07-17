from collections import defaultdict

class Execution_Graph():
    def __init__(self, source_tag = "source"):
        self.source_tag = source_tag
        
        self.tags_set = set([source_tag])
        
        self.completion_dict = dict()
        self.completion_dict[source_tag] = True
        
        self.transformer_dict = dict()
        self.transformer_dict[source_tag] = None
        
        self.desc_dict = defaultdict(set)
        self.asc_dict = defaultdict(set)
        
        self.desc_dict["source"].update()
        self.asc_dict["source"].update()
        
    def get_source_tag():
        return self.source_tag
    
    def add_step(self, step_tag, transformer, prerequisites = ["source"]):
        if step_tag in self.tags_set:
            raise Exception("Trying to rewrite step")
        
        self.tags_set.update([step_tag])
        
        self.completion_dict[step_tag] = False
        if transformer.tag == None:
            transformer.tag = step_tag
        self.transformer_dict[step_tag] = transformer
        
        self.desc_dict[step_tag].update()
        self.asc_dict[step_tag].update()
        
        for tag in prerequisites:
            if tag not in self.tags_set:
                raise Exception("Used prerequisite tag not defined")
            else:
                self.desc_dict[tag].update([step_tag])
                self.asc_dict[step_tag].update([tag])

    def remove_step(self, step_tag):
        if step_tag not in self.tags_set:
            raise Exception("Trying to remove undefined step {tag}".format(tag = step_tag))
            
        if len(self.desc_dict[step_tag]) != 0:
            raise Exception("Trying to remove step {tag} with nonzero amount of dependancies".format(tag = step_tag))

        for asc in self.asc_dict[step_tag]:
            self.desc_dict[asc].remove(step_tag)
        self.asc_dict.pop(step_tag)
        self.desc_dict.pop(step_tag)
        self.transformer_dict.pop(step_tag)
        self.completion_dict.pop(step_tag)
        self.tags_set.remove(step_tag)
        
    def set_completion(self, step_tag, completion = False):
        if step_tag not in self.tags_set:
            raise Exception("Trying to set completion for undefined tag")
        
        self.completion_dict[step_tag] = completion
        
    def get_transformer(self, step_tag):
        if step_tag not in self.tags_set:
            raise Exception("Trying to get transformer for undefined tag")
            
        return self.transformer_dict[step_tag]
    
    def transformers(self, step_tag):
        if not self.completion_dict[step_tag]:
            for ascendant in self.asc_dict[step_tag]:
                for transformer in self.transformers(ascendant):
                    yield transformer
            
            yield (step_tag, self.transformer_dict[step_tag])
            
    def all_transformers(self, step_tag):
        for ascendant in self.asc_dict[step_tag]:
            for transformer in self.transformers(ascendant):
                yield transformer
            
        yield (step_tag, self.transformer_dict[step_tag])
    
    def get_step_columns(self, step_tag):
        if step_tag not in self.tags_set:
            print "Trying to get columns for non existent step."
            return []
        return self.transformer_dict[step_tag].get_column_list()


class Model():
    def __init__(self, train, test, target):
        self.EG = Execution_Graph()
        
        self.train = train.copy()
        self.test = test.copy()
        
        self.target = target
    
    def add_step(self, *args, **kvargs):
        #TODO: Tracking dependancies based on feature list
        #TODO: Rework onehot encoder based on sklearn.DictVectoriser 
        #TODO: to make it possible to know all resultant columns beforehand
        #TODO: Make sure to only make transformers that know its columns beforehand
        self.EG.add_step(*args, **kvargs)

    def remove_step(self, tag):
        if tag not in self.EG.tags_set:
            print "Trying to remove undefined step {tag}.".format(tag = tag)
            return

        columns = set(self.EG.get_step_columns(tag))
        train_del = list(columns.intersection(set(self.train.columns)))
        test_del = list(columns.intersection(set(self.test.columns)))
        self.train.drop(train_del, axis = 1, inplace = True)
        self.test.drop(test_del, axis = 1, inplace = True)
        self.EG.remove_step(tag)
        
    def compute_step(self, step_tag):
        for cur_tag, transformer in self.EG.transformers(step_tag):
            try:
                new_train = transformer.fit_transform(self.train)
                new_test = transformer.transform(self.test)
            except:
                print ("".join(["Error in '", cur_tag, "' step, execution aborted"]))
                raise
                
            self.train = new_train
            self.test = new_test
            
            self.EG.set_completion(cur_tag, True)  

    def get_step_columns(self, step_tag):
        if isinstance(step_tag, list):
            fl = []
            for tag in step_tag:
                fl.extend(self.get_step_columns(tag))
            return list(set(fl))
        elif isinstance(step_tag, basestring):
            return self.EG.get_step_columns(step_tag)
        else:
            raise Exception("Step_tag must be either string or list of strings")