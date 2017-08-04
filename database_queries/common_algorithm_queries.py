"""
Contains queries used by various algorithms to gather data.
"""
from sqlalchemy import select, and_, exists, not_, func
from sqlalchemy.sql.expression import intersect_all

import database_interface.model as model
from database_interface.query.data_query import (
    BaseTransactionQuery, BaseTransactionResultQuery
)
from database_interface.query.internals_query import BaseTransactionMonitoringQuery


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
    

class TransactionQuery(BaseTransactionQuery):
    
    @classmethod
    def all_transaction_and_bank_base_columns(self):
        return (
            model.Transaction.pk,
            model.Transaction.amount,
            model.Transaction.operation_code,
            model.Transaction.operation_date,
            model.Transaction.bank_wording,
            model.Transaction.creditor_id,
            model.BankAccount.bank_code
        )

    @classmethod
    def all_company_usefull_columns(self):
        return (
            model.Company.activity_start_date,
            model.Company.ape_code_pk,
            model.Company.vat_regime_pk,
            model.Company.fiscal_regime_pk,
            model.Company.imposition_regime_pk,
            model.Company.bnc_option_pk,
            model.Company.legal_obligation_pk,
        )
    # @classmethod
    # def advanced_transaction_column(self):
    #     return (model.Transaction.creditor_id,)

    @classmethod
    def advanced_transaction_aggregation(cls):
        return cls.complement_wording_construct()
    
    @classmethod
    def all_results_columns(self):
        return (
            model.TransactionResult.account_number,
            model.TransactionResult.vat_type_pk,
            model.TransactionResult.vat_intra_flag_pk,
            model.TransactionResult.vat_class_pk
        )

    @classmethod
    def empty_transaction_factory(self):
        return GreatBigTransactionFactory()

    @classmethod
    def transaction_base(cls, condition_pk=True):
        bgf = GreatBigTransactionFactory()
        bgf.add_columns(cls.all_transaction_and_bank_base_columns())
        bgf.add_where_clause(condition_pk)
        return bgf.get_query()

    @classmethod
    def transaction_features(cls, condition_pk=True):
        bgf = GreatBigTransactionFactory()
        bgf.add_columns(cls.all_transaction_and_bank_base_columns())
        bgf.add_columns(cls.all_company_usefull_columns())
        bgf.add_select_clause(cls.advanced_transaction_aggregation)
        bgf.add_where_clause(condition_pk)
        return bgf.get_query()

    @classmethod
    def base_training_readable(cls, condition_pk=True):
        # :TODO: test if this query is faster in production environment
        # query = select(
        #     [model.Transaction.pk]
        # ).select_from(
        #     model.Transaction.__table__.outerjoin(
        #         model.TransactionMonitoring,
        #         and_(
        #             cls.transaction_monitoring_join(),
        #             model.TransactionMonitoring.algorithm_pk == algorithm_pk,
        #             model.TransactionMonitoring.active_flag == 1,
        #             model.TransactionMonitoring.status_id == model.TransactionMonitoring.SELECTED,
        #         )
        #     )
        # ).where(
        #     model.TransactionMonitoring.pk == None,
        # )
        """
        model.TransactionResult.vat_type_pk,
        model.TransactionResult.vat_intra_flag_pk,
        model.TransactionResult.vat_class_pk,
        """
        return select([
            model.Transaction.pk,
            model.Transaction.amount,
            model.Transaction.operation_code,
            model.Transaction.operation_date,
            model.Transaction.bank_wording,
            model.Transaction.creditor_id,
            model.BankAccount.bank_code,
            model.ApeCode.attribute,
            model.VatRegime.attribute,
            model.FiscalRegime.attribute,
            model.ImpositionRegime.attribute,
            model.BncOption.attribute,
            model.LegalObligation.attribute,
            model.TransactionResult.account_number,
            model.Company.activity_start_date,
            cls.complement_wording_construct(),
            ]).where(and_(
                condition_pk,
                cls.transaction_monitoring_join(),
                model.TransactionMonitoring.active_flag == 1,
                model.TransactionMonitoring.status_id == model.TransactionMonitoring.SELECTED,
                cls.transaction_company_join(),
                model.Company.bnc_option_pk == model.BncOption.pk,
                model.Company.ape_code_pk == model.ApeCode.pk,
                model.Company.vat_regime_pk == model.VatRegime.pk,
                model.Company.fiscal_regime_pk == model.FiscalRegime.pk,
                model.Company.imposition_regime_pk == model.ImpositionRegime.pk,
                model.Company.legal_obligation_pk == model.LegalObligation.pk,
                model.TransactionResult.algorithm_pk == 3,
                cls.transaction_result_join(),
                cls.bank_transaction_join(),
                )
            )

        
    @classmethod
    def transaction_base_training(cls, condition_pk=True):
        bgf = GreatBigTransactionFactory()
        bgf.add_columns(cls.all_transaction_and_bank_base_columns())
        bgf.add_columns(cls.all_company_usefull_columns())
        bgf.add_select_clause(cls.advanced_transaction_aggregation)
        bgf.add_columns(cls.all_results_columns())
        bgf.add_where_clause(condition_pk)
        return bgf.get_query()

    @classmethod
    def transaction_training(cls, condition_pk=True):
        bgf = GreatBigTransactionFactory()
        bgf.add_columns(cls.all_transaction_and_bank_base_columns())
        bgf.add_columns(cls.all_results_columns())
        bgf.add_where_clause(condition_pk)
        return bgf.get_query()

    @classmethod
    def count_valid_lines(cls):
        return select([func.count()]).where(model.Transaction.active_flag == 1)

    @classmethod
    def get_date_split(cls, number_of_training_lines):
        bgf = GreatBigTransactionFactory()
        bgf.add_column(model.Transaction.operation_date)
        bgf.add_where_clause(model.Transaction.active_flag == 1)
        bgf.order_by_column = model.Transaction.operation_date.asc()
        return bgf.get_query().offset(number_of_training_lines).limit(10)

    @classmethod
    def select_transaction_pks(cls, date_split, time):
        if time == 'before':
            condition_pk = model.Transaction.operation_date < date_split
        elif time == 'after':
            condition_pk = model.Transaction.operation_date >= date_split
        else:
            print('wrong time : {}'.format(time))
            return
        
        bgf = GreatBigTransactionFactory()
        bgf.add_column(model.Transaction.pk)
        bgf.add_where_clause(model.Transaction.active_flag == 1)
        bgf.add_where_clause(condition_pk)
        return bgf.get_query()

    @classmethod
    def transaction_not_selected_algorithm(cls, algorithm_pk):
        # :TODO: test if this query is faster in production environment
        # query = select(
        #     [model.Transaction.pk]
        # ).select_from(
        #     model.Transaction.__table__.outerjoin(
        #         model.TransactionMonitoring,
        #         and_(
        #             cls.transaction_monitoring_join(),
        #             model.TransactionMonitoring.algorithm_pk == algorithm_pk,
        #             model.TransactionMonitoring.active_flag == 1,
        #             model.TransactionMonitoring.status_id == model.TransactionMonitoring.SELECTED,
        #         )
        #     )
        # ).where(
        #     model.TransactionMonitoring.pk == None,
        # )
        return select([model.Transaction.pk]).where(
            not_(exists(select([model.TransactionMonitoring.pk]).where(and_(
                model.TransactionMonitoring.algorithm_pk == algorithm_pk,
                cls.transaction_monitoring_join(),
                model.TransactionMonitoring.active_flag == 1,
                model.TransactionMonitoring.status_id == model.TransactionMonitoring.SELECTED
            ))))
        )

    @classmethod
    def base_select_all_transactions_and_results(cls, condition_pk=True):
        return select([
            model.Transaction.pk,
            model.TransactionResult.account_number,
            model.TransactionResult.vat_type_pk,
            model.Transaction.amount,
            model.Transaction.operation_code,
            model.Transaction.operation_date,
            model.Transaction.bank_wording,
            model.BankAccount.bank_code,
            model.TransactionResult.vat_intra_flag_pk,
        ]).where(and_(
            cls.transaction_result_join(),
            TransactionResultQuery.ground_truth_result(),
            cls.bank_transaction_join(),
            condition_pk
        )).order_by(model.Transaction.pk.asc())

    @classmethod
    def base_select_all_partial_transactions(cls, condition_pk=True):
        return select([
            model.Transaction.pk,
            model.Transaction.amount,
            model.Transaction.operation_code,
            model.Transaction.operation_date,
            model.Transaction.bank_wording,
            model.BankAccount.bank_code
        ]).where(and_(
            cls.bank_transaction_join(),
            condition_pk
        )).order_by(model.Transaction.pk.asc())

    @classmethod
    def complement_wording_construct(cls):
        return func.concat_ws(
            ' ',
            func.array_to_string(model.Transaction.lib_wording, ' ', ''),
            model.Transaction.debtor_name, model.Transaction.debtor_id,
            model.Transaction.creditor_name, model.Transaction.creditor_id,
            model.Transaction.creditor_type, model.Transaction.ultimate_debtor_name,
            model.Transaction.ultimate_debtor_id, model.Transaction.ultimate_debtor_name,
            model.Transaction.ultimate_debtor_id, model.Transaction.remittance_information_1,
            model.Transaction.remittance_information_2, model.Transaction.creditor_reference_information,
            model.Transaction.creditor_account, model.Transaction.creditor_destinary_bank,
            model.Transaction.country_iso_code, model.Transaction.country_name, model.Transaction.origin_currency,
            func.array_to_string(model.Transaction.other_complements, ' ', ''),
        ).label('complement_wording')

    @classmethod
    def select_all_transaction_with_complements_and_results(cls, condition_pk=True):
        return select([
            model.Transaction.pk,
            model.TransactionResult.account_number,
            model.TransactionResult.vat_type_pk,
            model.Transaction.amount,
            model.Transaction.operation_code,
            model.Transaction.operation_date,
            model.Transaction.bank_wording,
            model.BankAccount.bank_code,
            model.Company.ape_code_pk,
            model.Company.vat_regime_pk,
            model.Company.fiscal_regime_pk,
            model.Company.imposition_regime_pk,
            cls.complement_wording_construct()
        ]).where(and_(
            cls.transaction_company_join(),
            cls.transaction_result_join(),
            cls.bank_transaction_join(),
            TransactionResultQuery.ground_truth_result(),
            condition_pk
        )).order_by(model.Transaction.pk.asc())

    @classmethod
    def select_all_transactions_aggregated(cls, condition_pk=True):
        return cls.select_all_transaction_with_complements_and_results(condition_pk)

    @classmethod
    def select_all_transaction_with_complements_and_no_results(cls, condition_pk=True):
        return select([
            model.Transaction.pk,
            model.Transaction.amount,
            model.Transaction.operation_code,
            model.Transaction.operation_date,
            model.Transaction.bank_wording,
            model.BankAccount.bank_code,
            model.Company.ape_code_pk,
            model.Company.vat_regime_pk,
            model.Company.fiscal_regime_pk,
            model.Company.imposition_regime_pk,
            cls.complement_wording_construct()
        ]).where(and_(
            cls.transaction_company_join(),
            cls.bank_transaction_join(),
            condition_pk
        )).order_by(model.Transaction.pk.asc())

    @classmethod
    def select_all_partial_transactions_aggregated(cls, condition_pk=True):
        return cls.select_all_transaction_with_complements_and_no_results(condition_pk)

    @classmethod
    def select_all_transaction_with_complements_and_results_for_tango(cls, condition_pk=True):
        return select([
            model.Transaction.pk,
            model.TransactionResult.account_number,
            model.TransactionResult.vat_type_pk,
            model.TransactionResult.vat_class_pk,
            model.Transaction.amount,
            model.Transaction.operation_code,
            model.Transaction.operation_date,
            model.Transaction.bank_wording,
            model.Transaction.creditor_id,
            model.BankAccount.bank_code,
            model.Company.ape_code_pk,
            model.Company.vat_regime_pk,
            model.Company.fiscal_regime_pk,
            model.Company.imposition_regime_pk,
            model.Company.activity_start_date,
            cls.complement_wording_construct()
        ]).where(and_(
            cls.transaction_company_join(),
            cls.transaction_result_join(),
            cls.bank_transaction_join(),
            TransactionResultQuery.ground_truth_result(),
            condition_pk
        )).order_by(model.Transaction.pk.asc())

    @classmethod
    def select_all_transaction_with_complements_and_no_results_for_tango(cls, condition_pk=True):
        return select([
            model.Transaction.pk,
            model.Transaction.amount,
            model.Transaction.operation_code,
            model.Transaction.operation_date,
            model.Transaction.bank_wording,
            model.Transaction.creditor_id,
            model.BankAccount.bank_code,
            model.Company.ape_code_pk,
            model.Company.vat_regime_pk,
            model.Company.fiscal_regime_pk,
            model.Company.imposition_regime_pk,
            model.Company.activity_start_date,
            cls.complement_wording_construct()
        ]).where(and_(
            cls.transaction_company_join(),
            cls.bank_transaction_join(),
            condition_pk
        )).order_by(model.Transaction.pk.asc())


class TransactionResultQuery(BaseTransactionResultQuery):

    @classmethod
    def select_all_results(cls, condition_algorithm, condition_id=True):
        return select([
            model.TransactionResult.transaction_pk,
            model.TransactionResult.algorithm_pk,
            model.TransactionResult.result_confidence,  # engine_confidence
            model.TransactionResult.account_number,
            model.TransactionResult.account_number_confidence,
            model.TransactionResult.vat_type_pk,
            model.TransactionResult.vat_type_confidence,
            model.TransactionResult.vat_class_pk,
            model.TransactionResult.vat_class_confidence,
            model.TransactionResult.vat_intra_flag_pk,
            model.TransactionResult.vat_intra_flag_confidence,
        ]).where(and_(
            cls.confidence_within_bounds(),
            condition_algorithm,
            condition_id
        )).order_by(model.TransactionResult.transaction_pk.asc())
    
    @classmethod
    def select_all_results_from_algorithm_list(cls, algorithm_pk_list=None):
        if algorithm_pk_list is None:
            condition_algorithm = model.TransactionResult.algorithm_pk >= 5
        else:
            condition_algorithm = model.TransactionResult.algorithm_pk.in_(algorithm_pk_list)
        return cls.select_all_results(condition_algorithm=condition_algorithm)
    
    @classmethod
    def select_engine_result_from_id_list(cls, id_list=None, algorithm_pk_list=None):
        
        if algorithm_pk_list is None:
            condition_algorithm = model.TransactionResult.algorithm_pk >= 5
        else:
            condition_algorithm = model.TransactionResult.algorithm_pk.in_(algorithm_pk_list)
        
        if id_list is None:
            condition_id = True
        else:
            condition_id = (model.TransactionResult.transaction_pk.in_(id_list))
        
        return cls.select_all_results(condition_algorithm=condition_algorithm, condition_id=condition_id)


class TransactionMonitoringQuery(BaseTransactionMonitoringQuery):
    @classmethod
    def select_transaction_pk_with_every_algorithm(cls, algorithm_pk_list):
        """
        Selects the distinct transaction pks with an active annotation for a
        every algorithm in the list.
        """
        return intersect_all(*(
            cls.select_algorithm_results_pk(algorithm_pk)
            for algorithm_pk in algorithm_pk_list
        ))
