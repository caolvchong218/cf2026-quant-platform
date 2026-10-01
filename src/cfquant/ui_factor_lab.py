"""Formula, measured evidence and optional model assistance in the existing UI."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import pandas as pd
import plotly.express as px
import streamlit as st
from .ui_design import heading, chart, note


def _read(path, default=None):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default


def experiment_folder(root):
    location=_read(root/'runs/research_v4_location.json',{})
    path=Path(location.get('output',root/'runs/research_v4'))
    return path if path.exists() else None


def save_proposals(root, answer):
    """Save only validated formula fields and model provenance, never credentials."""
    from .factor_lab import validate_expression
    folder=Path(root)/'runs/factor_plans';folder.mkdir(parents=True,exist_ok=True)
    saved=[]
    for proposal in answer.get('proposals',[]):
        expression=proposal['expression'];validate_expression(expression)
        ident=hashlib.sha256(expression.encode()).hexdigest()[:16]
        plan={k:proposal[k] for k in ('label','expression','hypothesis')}
        plan.update(id='custom_'+ident,source='model_research_proposal',
                    provider=answer.get('provider'),model=answer.get('model'),
                    prompt_version=answer.get('prompt_version'),status='dsl_validated_not_evaluated',
                    created_at=datetime.now(timezone.utc).isoformat())
        path=folder/(ident+'.json')
        if not path.exists():
            with path.open('x',encoding='utf-8') as stream:json.dump(plan,stream,ensure_ascii=False,indent=2)
        saved.append(plan['id'])
    return saved


def render(root):
    from .factor_lab import builtin_candidates,validate_expression
    from .llm_research import ResearchConnection,request_research,explain_holding
    root=Path(root)
    heading('因子研发实验室','提出公式、检验信号、留存候选，让每个研究结论都有来源。','AUXILIARY RESEARCH / V4')
    note('辅助路线：原年度 LightGBM 主线继续保留。模型负责研究与解释，持仓来自明确公式和可复算的组合规则。')
    formulas,evidence,explanation,connection=st.tabs(['公式与研究方案','真实数据对照','持仓解释','模型辅助'])
    holding_record=None
    with formulas:
        cards=_read(root/'evidence/research_v4/factor_cards.json',builtin_candidates())
        ids=[c['id'] for c in cards]
        chosen=st.selectbox('因子公式',ids,format_func=lambda x:next(c['label'] for c in cards if c['id']==x),key='lab_factor')
        card=next(c for c in cards if c['id']==chosen)
        st.code(card['expression'],language='text');st.write(card.get('hypothesis',''))
        st.caption('计算解释：只读取形成日及此前的数据；行业与市值中性化之后再进入研究组合。')
        with st.expander('编写新的表达式',expanded=False):
            expr=st.text_input('白名单公式','-rolling_std(returns(close, 1), 20)',key='lab_expression')
            hypothesis=st.text_input('经济假设','低波动信号可能改善风险调整收益。',key='lab_hypothesis')
            valid=False
            try:
                validate_expression(expr);valid=True;st.success('公式结构有效，尚未计算或验证收益。')
            except ValueError as exc:st.error(str(exc))
            if st.button('保存公式研究方案',disabled=not valid,key='lab_save'):
                ident=hashlib.sha256(expr.encode()).hexdigest()[:16]
                plan={'id':'custom_'+ident,'label':'自定义公式 '+ident[:6],'expression':expr,'hypothesis':hypothesis,
                      'source':'user_formula','created_at':datetime.now(timezone.utc).isoformat()}
                folder=root/'runs/factor_plans';folder.mkdir(parents=True,exist_ok=True)
                p=folder/(ident+'.json')
                if not p.exists():p.write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
                st.success('已保存方案；加入新实验后才产生实测结论。')
        with st.expander('运行新的研究实验'):
            from .research_lab import source_root,run_research
            data=source_root(root)
            available=(data/'data/private/research_v2/features.parquet').exists()
            st.caption('真实实验使用本地 Tushare 快照，输出独立目录。合成示例只检验软件流程。')
            include=st.checkbox('加入已保存的自定义公式',key='lab_include_custom')
            custom=[_read(p) for p in (root/'runs/factor_plans').glob('*.json')] if include else None
            if custom and len(custom)>20:st.warning('本轮最多加入20个自定义公式。');custom=custom[:20]
            if st.button('运行真实数据研究',disabled=not available,key='lab_real_run'):
                try:
                    with st.spinner('计算候选、年度训练及账本对照…'):
                        out=run_research(root,data_root=data,custom_candidates=custom)
                    st.session_state['lab_new_output']=str(out);st.success('新实验已保存：'+out.name)
                except (ValueError,FileExistsError,AssertionError) as exc:st.error(str(exc))
            if not available:st.info('当前电脑缺少私有快照，可查看随包真实聚合证据或运行合成流程。')
            if st.button('运行合成流程示例',key='lab_demo_run'):
                with st.spinner('验证研究流程…'):out=run_research(root,demo=True)
                st.success('合成流程已完成：'+out.name)
            if 'lab_new_output' in st.session_state:
                out=Path(st.session_state['lab_new_output'])
                st.dataframe(pd.read_csv(out/'candidates.csv'),hide_index=True,width='stretch')
    with evidence:
        path=root/'evidence/research_v4'
        if not (path/'candidates.csv').exists():st.info('真实对照结果尚未发布。')
        else:
            decision=_read(path/'decision.json',{})
            st.info('历史开发比较：既有区间已被观察，不能重新称为盲测。按验证期夏普预选辅助候选：'+decision.get('whitebox_preselected','—'))
            phase=st.radio('比较阶段',['historical','validation'],format_func=lambda x:'2025起历史对照' if x=='historical' else '2023–2024验证',horizontal=True,key='lab_phase')
            rows=pd.read_csv(path/'candidates.csv');rows=rows[rows.phase==phase]
            cols=['label','total_return','annualized_return','sharpe','max_drawdown','excess_vs_total_return','total_cost']
            display=rows[cols].copy()
            for name in ['total_return','annualized_return','max_drawdown','excess_vs_total_return']:display[name]*=100
            st.dataframe(display,hide_index=True,width='stretch',column_config={
                'label':'候选策略',
                'total_return':st.column_config.NumberColumn('累计收益（%）',format='%.2f'),
                'annualized_return':st.column_config.NumberColumn('年化收益（%）',format='%.2f'),
                'max_drawdown':st.column_config.NumberColumn('最大回撤（%）',format='%.2f'),
                'excess_vs_total_return':st.column_config.NumberColumn('超额全收益（百分点）',format='%.2f'),
                'total_cost':st.column_config.NumberColumn('累计成本（元）',format='%.2f'),
                'sharpe':st.column_config.NumberColumn('年化夏普',format='%.3f')})
            if phase=='historical':
                curves=[]
                for row in rows.itertuples():
                    p=path/'daily'/f'{row.candidate}.csv'
                    if p.exists():
                        frame=pd.read_csv(p,parse_dates=['date']);frame['净值']=frame.nav/1e6;frame['策略']=row.label;curves.append(frame)
                if curves:chart(px.line(pd.concat(curves),x='date',y='净值',color='策略'),430)
            st.download_button('下载候选对照表',rows.to_csv(index=False).encode('utf-8-sig'),'qingxu_candidate_comparison.csv')
            with st.expander('因子评价与训练身份'):
                st.dataframe(pd.read_csv(path/'factor_metrics.csv'),hide_index=True,width='stretch')
                st.json(_read(path/'protocol.json',{}))
    with explanation:
        folder=experiment_folder(root)
        p=folder/'holding_explanations.json' if folder else None
        if p and p.exists():
            records=_read(p,[])
            if not records:st.info('该实验末日没有股票持仓。')
            else:
                asset=st.selectbox('查看股票的信号贡献',[r['asset'] for r in records],key='lab_holding')
                record=next(r for r in records if r['asset']==asset)
                holding_record=record
                st.caption('信号形成日 '+record['date']+' · 辅助候选 '+record['candidate'])
                st.dataframe(pd.DataFrame(record['contributions']),hide_index=True,width='stretch')
                answer=explain_holding(record)
                st.write(answer.get('explanation',answer));st.caption(record['note'])
        else:st.info('持仓解释需要本地实验明细。随包公开证据只含聚合结果，补足快照后可重新运行。')
    with connection:
        st.write('模型根据研究问题、公式和明确提供的指标提出建议。数值成绩以评价器为准。')
        provider=st.selectbox('模型连接',['offline','openai_compatible','ollama'],format_func=lambda x:{'offline':'离线计算说明','openai_compatible':'Chat Completions 兼容接口','ollama':'本机 Ollama'}[x],key='lab_provider')
        model=st.text_input('模型名称','',key='lab_model')
        endpoint=st.text_input('服务地址','http://127.0.0.1:11434' if provider=='ollama' else 'https://api.openai.com/v1',key='lab_endpoint_'+provider)
        key=st.text_input('API 密钥',type='password',key='lab_api_key') if provider=='openai_compatible' else None
        question=st.text_area('研究问题','提出可解释的量价因子，并说明适用假设和可能失效的市场状态。',key='lab_question')
        max_output=st.number_input('输出上限（tokens）',128,4096,1024,key='lab_max_output')
        st.caption('一次点击发起一次请求，发送上述问题和当前公式；不发送 Token、原始逐股行情或整个实验目录。真实服务可能消耗你的模型额度。')
        if st.button('获取研究建议',key='lab_request',disabled=provider!='offline' and not model.strip()):
            try:
                spec=ResearchConnection(provider=provider,base_url=endpoint,model=model.strip() or 'deterministic',max_output_tokens=int(max_output))
                answer=request_research(spec,{'question':question,'factor':card,'scope':'research_and_explanation_only'},api_key=key)
                st.session_state['lab_model_answer']=answer
            except (ValueError,RuntimeError,OSError) as exc:st.error(str(exc))
        if holding_record is not None:
            st.caption('持仓解释只发送所选股票的形成日、公式与贡献，不发送完整行情。')
            if st.button('解释当前所选持仓',key='lab_explain_request',disabled=provider!='offline' and not model.strip()):
                try:
                    spec=ResearchConnection(provider=provider,base_url=endpoint,model=model.strip() or 'deterministic',max_output_tokens=int(max_output))
                    st.session_state['lab_model_answer']=explain_holding(holding_record,spec,api_key=key)
                except (ValueError,RuntimeError,OSError) as exc:st.error(str(exc))
        if 'lab_model_answer' in st.session_state:
            answer=st.session_state['lab_model_answer']
            st.caption(answer.get('explanation_label','模型返回'))
            st.write(answer.get('explanation',''))
            st.json(answer.get('proposals',[]))
            if answer.get('proposals') and st.button('将有效建议保存为研究方案',key='lab_save_proposals'):
                saved=save_proposals(root,answer)
                st.success(f'已保存 {len(saved)} 个待验证方案；在公式页勾选加入后可运行新的独立实验。')
            st.download_button('下载研究建议与来源',json.dumps(answer,ensure_ascii=False,indent=2),'factor_research_suggestion.json',mime='application/json')
