"""One place for course deliverables and a repeatable classroom walkthrough."""
import streamlit as st
from .ui_design import heading, note


def render(root):
    heading('材料与答辩', '把报告、演示和操作顺序放在一起，课堂上可以随时回到研究证据。')
    current=root/'reports/platform_v4'
    ready=(current/'CF2026_V4_Final_Report.pdf').exists()
    if not ready:st.info('平台增强版材料正在生成；下方保留已完成的 V3 材料。')
    folder=current if ready else root/'reports/benchmark_v3'
    prefix='CF2026_V4' if ready else 'CF2026_V3'
    tabs=st.tabs(['交付文件','现场演示','答辩准备'])
    with tabs[0]:
        for col,suffix,label,mime in zip(st.columns(3),['Final_Report.pdf','Beamer.pdf','Presentation.pptx'],
                ['最终报告','Beamer 演示','可编辑 PowerPoint'],['application/pdf','application/pdf','application/vnd.openxmlformats-officedocument.presentationml.presentation']):
            with col:
                st.markdown('**'+label+'**')
                p=folder/(prefix+'_'+suffix)
                if p.exists():st.download_button('下载'+label,p.read_bytes(),p.name,mime,key='materials_'+suffix,width='stretch')
                else:st.caption('文件准备中')
        version='V4' if ready else 'V3'
        method=root/'docs/SHARPE_RATIO.md'
        if method.exists():
            st.download_button('下载夏普比率说明',method.read_bytes(),method.name,'text/markdown')
        printable=folder/(prefix+'_Speaker_Script.pdf')
        if printable.exists():
            st.download_button('下载可打印演讲稿 PDF',printable.read_bytes(),printable.name,'application/pdf')
        script=root/f'docs/PRESENTATION_SCRIPT_{version}.md'
        if script.exists():
            st.download_button('下载逐页演讲稿',script.read_bytes(),script.name,'text/markdown')
            with st.expander('在线阅读讲稿'):st.markdown(script.read_text(encoding='utf-8'))
        for name,label in [('final_report.tex','报告 LaTeX 源码'),('beamer.tex','Beamer 源码')]:
            p=folder/name
            if p.exists():st.download_button(label,p.read_bytes(),name,'text/plain')
        st.caption('完整复现代码、图表源文件与依赖见 GitHub；本机提交包保存在桌面 GPT的输出 文件夹。')
        st.link_button('打开 GitHub 源码','https://github.com/caolvchong218/cf2026-quant-platform')
    with tabs[1]:
        steps=[('研究总览','说明 V3 的25.13%历史收益、9.88%回撤和历史重选边界。'),
               ('策略对比','选择 V3 与 V2，将基准设为全收益指数，缩短日期并解释区间切片。'),
               ('风险透镜','查看月收益，再找尚未恢复的回撤；移动现金配置滑块。'),
               ('因子诊断','切换低波动与中期动量，展示正负 IC 和有效样本。'),
               ('实验档案','选择一组实验，检索未完全成交订单，保存一句研究观察。'),
               ('Qlib 接入','展示真实调用结果、表达式与独立计算核对；说明是研究桥。')]
        for i,(page,body) in enumerate(steps,1):
            st.markdown(f'**{i}. {page}**');st.write(body)
        note('演示前先打开各页面并准备 PDF 备份。现场使用已有快照，模型重训与数据下载留到演示后；参数变更产生的结果应明确标为新实验。')
    with tabs[2]:
        qa={'跑赢指数说明模型有效吗？':'只能说明这段已观察历史的结果。V3经过历史重选，下一步必须冻结规则并用未来新数据验证。',
            '为什么还保留亏损策略？':'失败对照是判断改进来源的证据，也用于检查费用、调仓与信号是否被混淆。',
            'Qlib 接入后是否使用了Qlib的回测？':'请查看Qlib页的实际接入范围。因子计算、模型和成交引擎是不同层，不能把表达式接通说成整个框架已替换。',
            '现金滑块是不是重新回测？':'不是。它按固定比例分配原策略与零利息现金，是资金分配算术情景；完整重跑在回测实验页。',
            '账本正确是否等于策略好？':'不等于。账本核对资金与成交一致性，策略评估另外检查收益、风险、基准和样本外边界。'}
        for question,answer in qa.items():
            with st.expander(question):st.write(answer)
