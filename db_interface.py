import logging

from database_interface.database_interface import AlgorithmDatabaseInterface
import meta
import meta.algorithm_query as queries


OFFSET_FROM_TIIME = 0  # 10**11  # see database_interface.nine_gates_of_tiime.tiime_query OFFSET_FOR_TIIME
META_INPUT_ALGORITHM_CONFIG_KEY = 'meta_input_algorithms'
META_REQUIRED_ALGORITHM_CONFIG_KEY = 'meta_required_algorithms'


class MetaDatabaseInterface(AlgorithmDatabaseInterface):
    
    def __init__(self, uri):
        super(MetaDatabaseInterface, self).__init__(uri)
        
        # Creating the list of algorithms to use as input suggestion
        input_algorithm_name_set = set(meta.get_meta_config(META_INPUT_ALGORITHM_CONFIG_KEY))
        input_algorithm_name_set.add('meta')
        required_algorithm_name_set = set(meta.get_meta_config(META_REQUIRED_ALGORITHM_CONFIG_KEY))
        required_algorithm_name_set.add('meta')
        
        self.input_algorithm_pk_list = [self._register_algorithm_name(name) for name in input_algorithm_name_set]
        self.required_algorithms_pk_list = [self._register_algorithm_name(name) for name in required_algorithm_name_set]
        
        logging.info(
            'MetaDatabaseInterface init=done input_algorithm_pk_list=%s required_algorithms_pk_list=%s',
            self.input_algorithm_pk_list, self.required_algorithms_pk_list
        )
    
    # legacy function, not sure how useful they are, but it's not this sprint job to clean it
    def get_engines(self):
        """An ordered list of all available engines"""
        engines = [res[0] for res in self._get_query_results(queries.DistinctQuery.engines_query())]
        return sorted(set(engines))
    
    def get_accounts(self):
        """An ordered list of all available accounts"""
        return sorted(res[0] for res in self._get_query_results(queries.DistinctQuery.accounts_query())
                      if res[0] is not None)
        
    def get_intracom(self):
        """An ordered list of all available accounts"""
        return sorted(res[0] for res in self._get_query_results(queries.DistinctQuery.vat_intracom_query())
                      if res[0] is not None)
    
    def get_vat_codes(self):
        """An ordered list of all available vat codes"""
        vat_codes = set(res[0] for res in self._get_query_results(queries.DistinctQuery.vat_codes_query()))
        return sorted(vat_codes.union({5}))
    
    def get_training_inputs_auto_readable(self, set_name):
        """
        :return: an iterable with the inputs corresponding to the id_list
        """
        list_ids = self.engine.execute(
            queries.TrainSplittingQuery.get_values_split(set_name, split_id=None)).fetchone()[0]
        transaction_query = queries.select_transactions_info_from_id_list_readable(list_ids)
        
        engines_query = queries.TransactionResultQuery.select_all_results_from_algorithm_list(
            algorithm_pk_list=self.input_algorithm_pk_list)
    
        _transactions_info = dict()
        # meh, I have to find an other way to do that
        for result_line in self._get_query_results(transaction_query):
            transaction_id = result_line[0]
            _transactions_info[transaction_id] = result_line[1:]
        engines_results = self._get_query_results(engines_query)
    
        current_engines_results = None
        current_transaction_id = OFFSET_FROM_TIIME
        for engine_result_line in engines_results:
            transaction_id = engine_result_line[0]
            # Manage the case where we have a new transaction
            if current_transaction_id != transaction_id:
                # we want to get transactions without any engine results
                for no_engine_result_id in range(current_transaction_id + 1, transaction_id):
                    if no_engine_result_id in _transactions_info:
                        yield (no_engine_result_id, *_transactions_info[no_engine_result_id], [])
            
                if current_transaction_id in _transactions_info:
                    yield (
                        current_transaction_id, *_transactions_info[current_transaction_id],
                        current_engines_results)
            
                current_transaction_id = transaction_id
                current_engines_results = list()
        
            engine_str = engine_result_line[1]
            engine_result = tuple((engine_str, *engine_result_line[2:]))
            current_engines_results.append(engine_result)
    
        if current_transaction_id in _transactions_info:
            yield tuple(
                (current_transaction_id, *_transactions_info[current_transaction_id], current_engines_results))
    
    def get_training_inputs(self):
        """
        :return: an iterable with the inputs corresponding to the id_list
        """
        transaction_query = queries.TransactionQuery.select_all_transactions()
        engines_query = queries.TransactionResultQuery.select_all_results_from_algorithm_list(
            algorithm_pk_list=self.input_algorithm_pk_list)
        
        _transactions_info = dict()
        # meh, I have to find an other way to do that
        for result_line in self._get_query_results(transaction_query):
            transaction_id = result_line[0]
            _transactions_info[transaction_id] = result_line[1:]
        engines_results = self._get_query_results(engines_query)
        
        current_engines_results = None
        current_transaction_id = OFFSET_FROM_TIIME
        for engine_result_line in engines_results:
            transaction_id = engine_result_line[0]
            # Manage the case where we have a new transaction
            if current_transaction_id != transaction_id:
                # we want to get transactions without any engine results
                for no_engine_result_id in range(current_transaction_id + 1, transaction_id):
                    if no_engine_result_id in _transactions_info:
                        yield (no_engine_result_id, *_transactions_info[no_engine_result_id], [])
                
                if current_transaction_id in _transactions_info:
                    yield (
                        current_transaction_id, *_transactions_info[current_transaction_id],
                        current_engines_results)
                
                current_transaction_id = transaction_id
                current_engines_results = list()
            
            engine_str = engine_result_line[1]
            engine_result = tuple((engine_str, *engine_result_line[2:]))
            current_engines_results.append(engine_result)
        
        if current_transaction_id in _transactions_info:
            yield tuple(
                (current_transaction_id, *_transactions_info[current_transaction_id], current_engines_results))

    def get_training_inputs_auto(self, set_name):
        """
        :return: an iterable with the inputs corresponding to the id_list
        """
        list_ids = self.engine.execute(
            queries.TrainSplittingQuery.get_values_split(set_name, split_id=None)).fetchone()[0]
        transaction_query = queries.TransactionQuery.select_transactions_info_from_id_list(list_ids)
        
        engines_query = queries.TransactionResultQuery.select_all_results_from_algorithm_list(
            algorithm_pk_list=self.input_algorithm_pk_list)
    
        _transactions_info = dict()
        # meh, I have to find an other way to do that
        for result_line in self._get_query_results(transaction_query):
            transaction_id = result_line[0]
            _transactions_info[transaction_id] = result_line[1:]
        engines_results = self._get_query_results(engines_query)
    
        current_engines_results = None
        current_transaction_id = OFFSET_FROM_TIIME
        for engine_result_line in engines_results:
            transaction_id = engine_result_line[0]
            # Manage the case where we have a new transaction
            if current_transaction_id != transaction_id:
                # we want to get transactions without any engine results
                for no_engine_result_id in range(current_transaction_id + 1, transaction_id):
                    if no_engine_result_id in _transactions_info:
                        yield (no_engine_result_id, *_transactions_info[no_engine_result_id], [])
            
                if current_transaction_id in _transactions_info:
                    yield (
                        current_transaction_id, *_transactions_info[current_transaction_id],
                        current_engines_results)
            
                current_transaction_id = transaction_id
                current_engines_results = list()
        
            engine_str = engine_result_line[1]
            engine_result = tuple((engine_str, *engine_result_line[2:]))
            current_engines_results.append(engine_result)
    
        if current_transaction_id in _transactions_info:
            yield tuple(
                (current_transaction_id, *_transactions_info[current_transaction_id], current_engines_results))

    def get_transaction_query(self):
        return queries.TransactionQuery.select_all_transactions()
        
    def get_partial_transaction_query(self):
        return queries.TransactionQuery.select_all_partial_transactions()
    
    def get_engine_query(self):
        return queries.TransactionResultQuery.select_all_results_from_algorithm_list()
    
    def get_algorithm_inputs(self, id_list):
        """
        :return: an iterable with the inputs corresponding to the id_list
        :TODO: Rewrite the function - Potentially BUGGED
        """
        transaction_query = queries.TransactionQuery.select_partial_transactions_info_from_id_list(id_list)
        logging.warning('Test')
        logging.warning(transaction_query)
        logging.warning(id_list)
        engines_query = queries.TransactionResultQuery.select_engine_result_from_id_list(
            id_list, algorithm_pk_list=self.input_algorithm_pk_list)
        _transactions_info = dict()
        # meh, I have to find an other way to do that
        for result_line in self._get_query_results(transaction_query):
            transaction_id = result_line[0]
            _transactions_info[transaction_id] = result_line[1:]
        engines_results = self._get_query_results(engines_query)
        
        current_engines_results = None
        current_transaction_id = OFFSET_FROM_TIIME
        for engine_result_line in engines_results:
            transaction_id = engine_result_line[0]
            
            # Manage the case where we have a new transaction
            if current_transaction_id != transaction_id:
                # we want to get transactions without any engine results
                for no_engine_result_id in range(current_transaction_id + 1, transaction_id):
                    if no_engine_result_id in _transactions_info:
                        yield (no_engine_result_id, *_transactions_info[no_engine_result_id], [])
            
                if current_transaction_id in _transactions_info:
                    yield (
                        current_transaction_id, *_transactions_info[current_transaction_id],
                        current_engines_results)
                
                current_transaction_id = transaction_id
                current_engines_results = list()
            
            engine_str = engine_result_line[1]
            engine_result = tuple((engine_str, *engine_result_line[2:]))
            current_engines_results.append(engine_result)
        
        if current_transaction_id in _transactions_info:
            yield tuple(
                (current_transaction_id, *_transactions_info[current_transaction_id], current_engines_results))
    
    def get_id_inputs(self, algorithm_pk, algorithm_version_pk):
        """
        Get the pks of the transactions to annotate, this is used by the daimyo.
        
        :return: an iterable of transaction_pk
        
        :TODO: see if we need to use the version: algorithm_version_pk
        """
        fully_annotated_pk_query = queries.TransactionMonitoringQuery.\
            select_transaction_pk_with_every_algorithm(self.required_algorithms_pk_list)
        
        meta_unannotated_query = queries.TransactionQuery.\
            transaction_not_selected_algorithm(algorithm_pk)
        
        fully_annotated_pk = self._get_query_single_result(fully_annotated_pk_query)
        meta_unannotated = self._get_query_single_result(meta_unannotated_query)
        
        return set(fully_annotated_pk).intersection(meta_unannotated)
    
