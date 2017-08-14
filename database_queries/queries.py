from sqlalchemy import and_, or_, select, distinct, func

from database_interface import model, query
from data_viz.database_queries import common_algorithm_queries

"""
class TransactionQuery(BaseTransactionQuery):
    @classmethod
    def transaction_base_training(cls, condition_pk=True):
        bgf = GreatBigTransactionFactory()
        bgf.add_columns(cls.all_transaction_and_bank_base_columns())
        bgf.add_columns(cls.all_company_usefull_columns())
        bgf.add_select_clause(cls.advanced_transaction_aggregation)
        bgf.add_columns(cls.all_results_columns())
        bgf.add_where_clause(condition_pk)
        return bgf.get_query()
"""


class GreatBigTransactionFactory:
    """
    Prepares simple queries of the form 'select X where Y order_by Z'
    
    Queries can be retrieved using `get_query`.
    """
    def __init__(self, columns=None):
        self.select_clause = []
        self.where_clause = []
        # models are identified by their tablename
        self.present_tables = {model.Transaction.__tablename__}
        self.order_by_column = model.Transaction.pk.asc()

        self.join_dict = {
            model.TransactionResult.__tablename__: (
                BaseTransactionQuery.transaction_result_join(),
                BaseTransactionResultQuery.ground_truth_result(),),
            model.Company.__tablename__: (
                BaseTransactionQuery.transaction_company_join(),),
            model.BankAccount.__tablename__: (
                BaseTransactionQuery.bank_transaction_join(),),
        }
        
        if columns:
            self.add_columns(columns)
        
    def get_query(self):
        return select([
            *tuple(self.select_clause)
        ]).where(and_(
            *tuple(self.where_clause)
        )).order_by(self.order_by_column)
    
    def add_model(self, model_table_name):
        for join_clause in self.join_dict.get(model_table_name):
            self.where_clause.append(join_clause)
        self.present_tables.update([model_table_name])
        
    def add_column(self, column_object):
        column_table = column_object.class_.__tablename__
        if column_table not in self.present_tables:
            self.add_model(column_table)
        self.select_clause.append(column_object)
    
    def add_columns(self, tuple_commands):
        for column in tuple_commands:
            self.add_column(column)
    
    def add_select_clause(self, column_function):
        self.select_clause.append(column_function())

    def add_where_clause(self, condition):
        self.where_clause.append(condition)
    

class TransactionResultQuery(query.TransactionResultQuery):
    pass


class TransactionQuery(common_algorithm_queries.TransactionQuery):

    @classmethod
    def select_all_transactions(cls, condition_id=True):
        return cls.transaction_base_training(condition_id)
    
    @classmethod
    def select_all_transactions_readable(cls, condition_id=True):
        return cls.base_training_readable(condition_id)

    @classmethod
    def select_all_partial_transactions(cls, condition_id=True):
        return cls.transaction_features(condition_id)
        
    @classmethod
    def select_transactions_to_annote_id(cls, algorithm_pk, waiting_algorithms_pk):
        return select([
            model.TransactionMonitoring.transaction_pk.label('pk'),
        ]).where(and_(
            or_(
                and_(
                    model.TransactionMonitoring.algorithm_pk.in_(waiting_algorithms_pk),
                    model.TransactionMonitoring.status_id == model.TransactionMonitoring.DONE,
                ).self_group(),
                model.TransactionMonitoring.algorithm_pk == algorithm_pk,
            ),
            model.TransactionMonitoring.active_flag == 1,
        )).group_by(
            model.TransactionMonitoring.transaction_pk,
        ).having(
            and_(
                func.count(model.TransactionMonitoring.algorithm_pk.distinct()) == len(waiting_algorithms_pk),
                func.min(model.TransactionMonitoring.algorithm_pk) != algorithm_pk
            ),
        )
    
    @classmethod
    def select_transactions_info_from_id_list(cls, id_list):
        # that's pretty dirty, we should find a way to extract the id_list from the query
        condition_id = (model.Transaction.pk.in_(id_list))
        return cls.select_all_transactions(condition_id=condition_id)

    @classmethod
    def select_transactions_info_from_id_list_readable(cls, id_list):
        condition_id = (model.Transaction.pk.in_(id_list))
        return cls.select_all_transactions_readable(condition_id=condition_id)

    @classmethod
    def select_partial_transactions_info_from_id_list(cls, id_list):
        # that's pretty dirty, we should find a way to extract the id_list from the query
        condition_id = (model.Transaction.pk.in_(id_list))
        return cls.select_all_partial_transactions(condition_id=condition_id)


class TransactionMonitoringQuery(query.TransactionMonitoringQuery):
    pass


class DistinctQuery(object):
    @classmethod
    def engines_query(cls):
        return select([distinct(model.AlgorithmName.pk)]).where(model.AlgorithmName.pk >= 5)

    @classmethod
    def accounts_query(cls):
        return select([distinct(model.TransactionResult.account_number)])

    @classmethod
    def vat_codes_query(cls):
        return select([distinct(model.VatType.pk)])
    
    @classmethod
    def vat_intracom_query(cls):
        return select([distinct(model.VatIntraFlag.pk)])


class AlgorithmNameQuery(query.AlgorithmNameQuery):
    pass


class TrainSplittingQuery(query.TrainSplittingQuery):
    pass
