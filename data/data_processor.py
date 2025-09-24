import json
import pandas as pd
from typing import List, Dict, Any
import os
from tqdm import tqdm
from io import StringIO

def process_jsonl_to_dataframe(data_path: str) -> pd.DataFrame:
    # 统计总行数
    if not os.path.exists(data_path):
        print(f"Error: File not found at {data_path}. Please check the file path.")
        return pd.DataFrame()
    
    with open(data_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    print(f"Total lines to process: {total_lines}")

    data_list = []
    buffer = StringIO()  
    open_brackets = 0 

    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            for i, line in tqdm(enumerate(f, 1), total=total_lines, desc="Processing JSONL"):
                buffer.write(line)
                # 统计括号以判断 JSON 完整性
                for char in line:
                    if char == '{':
                        open_brackets += 1
                    elif char == '}':
                        open_brackets -= 1
                
                # 当括号平衡（open_brackets == 0）且有内容时尝试解析
                if open_brackets == 0 and buffer.getvalue().strip():
                    try:
                        json_data = json.loads(buffer.getvalue())
                        loan_cases = json_data.get('data', {}).get('loanCases', [])
                        for case in loan_cases:
                            data_list.append(case)
                        buffer.truncate(0)  # 清空缓冲
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping malformed block at line {i}: {buffer.getvalue()}")
                        buffer.truncate(0)  # 清空损坏的数据
                elif open_brackets < 0:  # 意外的 }，清空缓冲
                    print(f"Warning: Unexpected closing bracket at line {i}: {buffer.getvalue()}")
                    buffer.truncate(0)
                    open_brackets = 0

        print(f"Finished processing {i} lines.")
    except FileNotFoundError:
        print(f"Error: File not found at {data_path}. Please check the file path.")
        return pd.DataFrame()

    if not data_list:
        print("No data found in the file.")
        return pd.DataFrame()

    processed_data = []
    for case in data_list:
        customers_list = case.get('customers', [])
        customer = customers_list[0] if customers_list and isinstance(customers_list, list) else {}
        employment = customer.get('employments', [{}])[0] if customer.get('employments') and isinstance(customer.get('employments'), list) else {}
        address = customer.get('addresses', [{}])[0] if customer.get('addresses') and isinstance(customer.get('addresses'), list) else {}
        
        tu_report_t1 = case.get('tuReport', {}).get('T1')
        
        state = case.get('state', {})
        state_code = state.get('code')
        state_name = state.get('name')
        
        if state_code == 2900 or state_name == "Writeoff":
            loanAmountApplied = totalTenorApplied = flatRateApplied = None
            loanAmountApproved = totalTenorApproved = flatRateApproved = None
        elif state_name == "Completed":
            applied_plan = case.get('appliedPlan', {})
            approval_plan = case.get('approvalPlan', {})
            loanAmountApplied = applied_plan.get('loanAmount')
            totalTenorApplied = applied_plan.get('totalTenor')
            flatRateApplied = applied_plan.get('flatRate')
            loanAmountApproved = approval_plan.get('loanAmount')
            totalTenorApproved = approval_plan.get('totalTenor')
            flatRateApproved = approval_plan.get('flatRate')
        else:
            applied_plan = case.get('appliedPlan', {})
            approval_plan = case.get('approvalPlan', {})
            loanAmountApplied = applied_plan.get('loanAmount')
            totalTenorApplied = applied_plan.get('totalTenor')
            flatRateApplied = applied_plan.get('flatRate')
            loanAmountApproved = approval_plan.get('loanAmount')
            totalTenorApproved = approval_plan.get('totalTenor')
            flatRateApproved = approval_plan.get('flatRate')

        row_data = {
            'age': customer.get('age'),
            'gender': customer.get('gender'),
            'propertyType': address.get('propertyType', {}).get('name') if isinstance(address.get('propertyType'), dict) else address.get('propertyType'),
            'workDurationYear': employment.get('workDurationYear'),
            'industry': employment.get('industry'),
            'employmentType': employment.get('employmentType'),
            'salaryAmount': employment.get('salaryAmount'),
            'preDsr': case.get('preDsr'),
            'postDsr': case.get('postDsr'),
            'scorePl01': tu_report_t1.get('scorePl01') if tu_report_t1 else None,
            'scoreCm01': tu_report_t1.get('scoreCm01') if tu_report_t1 else None,
            'stateCode': state_code,
            'stateName': state_name,
            'loanDate': case.get('loanDate'),
            'writeoffDate': case.get('writeoffDate'),
            'ocaDate': case.get('ocaDate'),
            'ivaDate': case.get('ivaDate'),
            'loanAmountApplied': loanAmountApplied,
            'totalTenorApplied': totalTenorApplied,
            'flatRateApplied': flatRateApplied,
            'loanAmountApproved': loanAmountApproved,
            'totalTenorApproved': totalTenorApproved,
            'flatRateApproved': flatRateApproved
        }
        processed_data.append(row_data)

    df = pd.DataFrame(processed_data)
    return df

def process_api_data(api_data: Dict[str, Any]) -> pd.DataFrame:
    identity_info = api_data.get('identity_info', {})
    credit_history = api_data.get('credit_history', {})
    bank_flow = api_data.get('bank_flow', {})
    applied_plan = api_data.get('appliedPlan', {})  # 新增：提取申请计划
    
    row_data = {
        'age': identity_info.get('age'),
        'gender': identity_info.get('gender'),
        'propertyType': identity_info.get('property_type', ''),
        'workDurationYear': identity_info.get('working_years'),
        'industry': identity_info.get('occupation_type', ''),
        'employmentType': '',  # 默认为空字符串
        'salaryAmount': bank_flow.get('monthly_avg_income'),
        'preDsr': bank_flow.get('preDsr'),
        'postDsr': bank_flow.get('postDsr'),
        # 从 appliedPlan 提取
        'loanAmountApplied': applied_plan.get('loanAmount'),
        'totalTenorApplied': applied_plan.get('totalTenor'),
        # 其他贷款字段仍为 None（API 无审批计划）
        'flatRateApplied': None,
        'loanAmountApproved': None,
        'totalTenorApproved': None,
        'flatRateApproved': None,
        # 信用历史
        'scorePl01': credit_history.get('pls_rating'),
        'scoreCm01': credit_history.get('cms_rating'),
        # 其他字段
        'stateCode': None,
        'loanDate': None,
        'writeoffDate': None,
        'ocaDate': None,
        'ivaDate': None
    }
    
    return pd.DataFrame([row_data])