# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# --- Configuração da Página ---
st.set_page_config(
    layout="wide",
    page_title="Dashboard Análise Manutenção de Máq e Equip | Real vs Orçado",
    page_icon="📊"
)

st.title('📊 Análise Manutenção de Máquinas e Equipamentos - Real vs Orçado')

# --- Carregamento e Pré-processamento dos Dados ---
@st.cache_data
def load_and_preprocess_data(filepath):
    try:
        df = pd.read_excel(filepath)

        # Converte 'Mês' para datetime e extrai ano/mês
        df['Mês'] = pd.to_datetime(df['Mês'], errors='coerce')
        df.dropna(subset=['Mês'], inplace=True)
        
        df['Ano'] = df['Mês'].dt.year
        df['Mês_Num'] = df['Mês'].dt.month # Mês numérico para ordenação
        df['Mês_Ano'] = df['Mês'].dt.strftime('%Y-%m')
        df['Nome_Mês'] = df['Mês'].dt.strftime('%b')  # Abreviação do mês (Jan, Fev, etc)
        
        # Garante que temos apenas dados de 2024 e 2025
        df = df[df['Ano'].isin([2024, 2025])]
        
        # Converte colunas numéricas
        for col in ['Real', 'Orçado']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Calcula variação e % variação
        df['Variação'] = df['Real'] - df['Orçado']
        df['% Variação'] = (df['Variação'] / df['Orçado']).replace([np.inf, -np.inf], np.nan).fillna(0) * 100
        
        return df
    except Exception as e:
        st.error(f"Erro ao carregar ou processar os dados: {e}")
        return pd.DataFrame()

# Define o caminho do arquivo de dados
# IMPORTANTE: Substitua pelo caminho real do seu arquivo Excel.
# Por exemplo, se estiver no mesmo diretório do seu script: filepath = "maqequip.xlsx"
filepath = "https://raw.githubusercontent.com/AlexandrePneuBras/AlexandrePneuBras-maq_e_equip/refs/heads/main/maqequip.xlsx"
df = load_and_preprocess_data(filepath)

if not df.empty:
    # --- Inicializa o Estado da Sessão para Filtros ---
    # Isso ajuda a gerenciar a funcionalidade "Selecionar Todos" e "Limpar Seleção"
    if 'selected_years' not in st.session_state:
        st.session_state.selected_years = sorted(df['Ano'].unique())
    if 'selected_branches' not in st.session_state:
        st.session_state.selected_branches = sorted(df['Cod. Filial'].astype(str).unique())
    if 'selected_segments' not in st.session_state:
        st.session_state.selected_segments = sorted(df['Segmento'].unique())
    if 'selected_months' not in st.session_state:
        # Obtém meses únicos e os ordena pela ordem numérica
        unique_months_df = df[['Mês_Num', 'Nome_Mês']].drop_duplicates().sort_values('Mês_Num')
        st.session_state.selected_months = list(unique_months_df['Nome_Mês'])
    if 'value_type_selection' not in st.session_state:
        st.session_state.value_type_selection = "Ambos" # Padrão para "Ambos"

    # --- Filtros da Barra Lateral ---
    st.sidebar.header("Filtros de Análise 🔎")
    st.sidebar.markdown("Use os filtros abaixo para refinar os dados exibidos no dashboard.")

    # Função auxiliar para criar filtro com botões de selecionar/limpar tudo
    def create_filter_with_buttons(label, options, session_state_key):
        st.sidebar.subheader(f"Filtrar por {label}")
        col_select_all, col_clear_all = st.sidebar.columns(2)

        with col_select_all:
            if st.button(f"Selecionar Todos", key=f"select_all_{session_state_key}"):
                st.session_state[session_state_key] = options
                st.rerun()
        with col_clear_all:
            if st.button(f"Limpar Seleção", key=f"clear_all_{session_state_key}"):
                st.session_state[session_state_key] = []
                st.rerun()

        current_selection = st.sidebar.multiselect(
            f"Escolha os {label.lower()}:",
            options=options,
            default=st.session_state[session_state_key],
            key=session_state_key
        )
        return current_selection

    # Filtro de Tipo de Valor (Real, Orçado, Ambos)
    st.sidebar.markdown("---")
    st.sidebar.subheader("Tipo de Valor para Gráficos 📊")
    value_type_selection = st.sidebar.radio(
        "Selecione o tipo de valor para visualização:",
        ("Ambos", "Real", "Orçado"),
        key='value_type_selection',
        help="Escolha se deseja ver 'Real', 'Orçado' ou 'Ambos' nos gráficos de comparação."
    )

    # Filtro de Ano
    st.sidebar.markdown("---")
    all_years = sorted(df['Ano'].unique())
    selected_years = create_filter_with_buttons("Ano", all_years, 'selected_years')
    
    # Filtro de Mês
    st.sidebar.markdown("---")
    unique_months_df = df[['Mês_Num', 'Nome_Mês']].drop_duplicates().sort_values('Mês_Num')
    all_months = list(unique_months_df['Nome_Mês'])
    selected_months = create_filter_with_buttons("Mês", all_months, 'selected_months')

    # Filtro de Filial
    st.sidebar.markdown("---")
    all_branches = sorted(df['Cod. Filial'].astype(str).unique())
    selected_branches = create_filter_with_buttons("Filiais", all_branches, 'selected_branches')
    
    # Filtro de Segmento
    st.sidebar.markdown("---")
    all_segments = sorted(df['Segmento'].unique())
    selected_segments = create_filter_with_buttons("Segmentos", all_segments, 'selected_segments')

    # Aplica os filtros
    filtered_df = df[
        df['Ano'].isin(selected_years) &
        df['Nome_Mês'].isin(selected_months) & # Aplica filtro de mês
        df['Cod. Filial'].astype(str).isin(selected_branches) &
        df['Segmento'].isin(selected_segments)
    ]

    # --- Métricas Resumo ---
    st.markdown("---")
    st.header("Indicadores Chave 📈")
    
    # Calcula os totais
    total_real = filtered_df['Real'].sum()
    total_orcado = filtered_df['Orçado'].sum()
    variacao_total = total_real - total_orcado
    perc_variacao = (variacao_total / total_orcado) * 100 if total_orcado != 0 else 0
    
    # Cria colunas para as métricas
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Real", f"R$ {total_real:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
    with col2:
        st.metric("Total Orçado", f"R$ {total_orcado:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
    with col3:
        st.metric("Variação Total", 
                  f"R$ {variacao_total:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."),
                  f"{perc_variacao:.2f}%")

    # --- Comparativo Anual ---
    st.markdown("---")
    st.header("Comparativo Anual: Real vs Orçado 📆")
    
    if len(selected_years) > 0:
        # Agrupa por ano
        yearly_comparison = filtered_df.groupby('Ano')[['Real', 'Orçado']].sum().reset_index()
        
        # Derrete para melhor plotagem com base na seleção do tipo de valor
        if value_type_selection == "Ambos":
            yearly_melted = yearly_comparison.melt(id_vars='Ano', 
                                                   value_vars=['Real', 'Orçado'],
                                                   var_name='Tipo', 
                                                   value_name='Valor')
            # Cria uma coluna combinada 'Tipo_Ano' para coloração distinta
            yearly_melted['Tipo_Ano'] = yearly_melted['Tipo'] + ' ' + yearly_melted['Ano'].astype(str)
        elif value_type_selection == "Real":
            yearly_melted = yearly_comparison[['Ano', 'Real']].rename(columns={'Real': 'Valor'})
            yearly_melted['Tipo'] = 'Real'
            yearly_melted['Tipo_Ano'] = yearly_melted['Tipo'] + ' ' + yearly_melted['Ano'].astype(str)
        else: # value_type_selection == "Orçado"
            yearly_melted = yearly_comparison[['Ano', 'Orçado']].rename(columns={'Orçado': 'Valor'})
            yearly_melted['Tipo'] = 'Orçado'
            yearly_melted['Tipo_Ano'] = yearly_melted['Tipo'] + ' ' + yearly_melted['Ano'].astype(str)

        # Define cores personalizadas para cada combinação de Tipo e Ano
        custom_colors = {
            'Real 2024': '#1f77b4',  # Azul para Real 2024
            'Orçado 2024': '#aec7e8', # Azul mais claro para Orçado 2024
            'Real 2025': '#2ca02c',  # Verde para Real 2025
            'Orçado 2025': '#98df8a'  # Verde mais claro para Orçado 2025
        }
        # Filtra o mapa de cores para incluir apenas as chaves realmente presentes nos dados
        filtered_colors = {k: v for k, v in custom_colors.items() if k in yearly_melted['Tipo_Ano'].unique()}

        # Cria gráfico de barras
        fig_yearly = px.bar(yearly_melted,
                            x='Ano',
                            y='Valor',
                            color='Tipo_Ano', # Usa a nova coluna combinada para coloração
                            barmode='group',
                            text='Valor',
                            labels={'Valor': 'Valor (R$)', 'Ano': 'Ano', 'Tipo_Ano': 'Tipo & Ano'},
                            height=500,
                            color_discrete_map=filtered_colors) # Aplica as cores personalizadas filtradas
        
        # Formata valores e layout
        fig_yearly.update_traces(texttemplate='R$ %{value:,.2f}', textposition='outside')
        fig_yearly.update_layout(hovermode='x unified',
                                 yaxis_tickprefix='R$ ',
                                 yaxis_tickformat=',.2f',
                                 legend_title_text='')
        st.plotly_chart(fig_yearly, use_container_width=True)
        
        # Comparativo Mensal
        st.subheader("Comparativo Mensal: Real vs Orçado 📈")
        
        # Agrupa por mês-ano
        monthly_comparison = filtered_df.groupby(['Ano', 'Mês_Num', 'Mês_Ano', 'Nome_Mês'])[['Real', 'Orçado']].sum().reset_index()
        monthly_comparison = monthly_comparison.sort_values(['Ano', 'Mês_Num'])
        
        # Cria gráfico de linhas
        fig_monthly = go.Figure()
        
        # Adiciona traces para cada ano com base na seleção do tipo de valor
        for year in selected_years:
            year_data = monthly_comparison[monthly_comparison['Ano'] == year]
            
            # Usa cores específicas com base no ano para Real e Orçado
            color_real = custom_colors.get(f'Real {year}', '#1f77b4') # Padrão para azul se não encontrado
            color_orcado = custom_colors.get(f'Orçado {year}', '#ff7f0e') # Padrão para laranja se não encontrado

            if value_type_selection in ["Ambos", "Real"]:
                fig_monthly.add_trace(go.Scatter(
                    x=year_data['Nome_Mês'],
                    y=year_data['Real'],
                    name=f'Real {year}',
                    mode='lines+markers',
                    line=dict(width=3, color=color_real), 
                    hovertemplate='<b>%{x}</b><br>Real: R$ %{y:,.2f}<extra></extra>'
                ))
            
            if value_type_selection in ["Ambos", "Orçado"]:
                fig_monthly.add_trace(go.Scatter(
                    x=year_data['Nome_Mês'],
                    y=year_data['Orçado'],
                    name=f'Orçado {year}',
                    mode='lines+markers',
                    line=dict(dash='dot', width=3, color=color_orcado), 
                    hovertemplate='<b>%{x}</b><br>Orçado: R$ %{y:,.2f}<extra></extra>'
                ))
        
        # Atualiza layout
        fig_monthly.update_layout(
            title='Evolução Mensal: Real vs Orçado',
            xaxis_title='Mês',
            yaxis_title='Valor (R$)',
            hovermode='x unified',
            yaxis_tickprefix='R$ ',
            yaxis_tickformat=',.2f',
            legend_title_text=''
        )
        st.plotly_chart(fig_monthly, use_container_width=True)
        
        # Análise de Variação
        st.subheader("Análise de Variação (Real - Orçado) 📊")
        
        # Calcula variação mensal
        monthly_comparison['Variação'] = monthly_comparison['Real'] - monthly_comparison['Orçado']
        # Recalcula % Variação com base nos dados agrupados mensalmente
        monthly_comparison['% Variação'] = (monthly_comparison['Variação'] / monthly_comparison['Orçado']).replace([np.inf, -np.inf], np.nan).fillna(0) * 100
        
        # Cria gráfico de variação (sempre mostra variação, não afetado pela seleção de tipo de valor)
        fig_variance = go.Figure()

        # Define cores específicas para variação com base no ano e sinal
        variance_colors_map = {
            '2024_Pos': '#33a02c', # Verde mais escuro para positivo 2024
            '2024_Neg': '#e31a1c', # Vermelho mais escuro para negativo 2024
            '2025_Pos': '#b2df8a', # Verde mais claro para positivo 2025
            '2025_Neg': '#f9c8c9'  # Vermelho mais claro para negativo 2025
        }
        
        for year in selected_years:
            year_data = monthly_comparison[monthly_comparison['Ano'] == year]
            
            # Determina cores para as barras com base no ano e sinal de variação
            bar_colors = []
            for val in year_data['Variação']:
                if year == 2024:
                    bar_colors.append(variance_colors_map['2024_Pos'] if val >= 0 else variance_colors_map['2024_Neg'])
                elif year == 2025:
                    bar_colors.append(variance_colors_map['2025_Pos'] if val >= 0 else variance_colors_map['2025_Neg'])
                else: # Fallback para qualquer outro ano, embora filtrado para 2024/2025
                    bar_colors.append('green' if val >= 0 else 'red')
            
            fig_variance.add_trace(go.Bar(
                x=year_data['Nome_Mês'],
                y=year_data['Variação'],
                name=str(year),
                text=year_data['% Variação'].apply(lambda x: f'{x:.1f}%'),
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Variação: R$ %{y:,.2f}<br>% Variação: %{text}<extra></extra>',
                marker_color=bar_colors # Aplica as cores específicas do ano
            ))
        
        fig_variance.update_layout(
            barmode='group',
            title='Variação Mensal: Real vs Orçado',
            xaxis_title='Mês',
            yaxis_title='Variação (R$)',
            hovermode='x unified',
            yaxis_tickprefix='R$ ',
            yaxis_tickformat=',.2f',
            height=600, # Aumenta a altura para evitar que os rótulos sejam cortados
            margin=dict(t=50) # Adiciona margem superior
        )
        st.plotly_chart(fig_variance, use_container_width=True)
        
    else:
        st.warning("Selecione pelo menos um ano para visualizar as comparações.")

    # --- Top Filiais por Custo ---
    st.markdown("---")
    st.header("Top Filiais com Maiores Custos por Ano 💰")

    if len(selected_years) > 0:
        # Entrada do usuário para o número de top filiais
        num_top_branches = st.slider(
            "Selecione a quantidade de top filiais para exibir:",
            min_value=1,
            max_value=min(len(df['Cod. Filial'].unique()), 100), # Máximo de 10 ou o número de filiais únicas se for menor
            value=min(5, len(df['Cod. Filial'].unique())) # Padrão para 5 ou menos se não houver filiais suficientes
        )

        top_branches_data = pd.DataFrame()
        for year in selected_years:
            # Filtra dados para o ano atual
            df_year = filtered_df[filtered_df['Ano'] == year]
            
            # Agrupa por 'Cod. Filial' e soma 'Real' e 'Orçado'
            branch_costs = df_year.groupby('Cod. Filial')[['Real', 'Orçado']].sum().reset_index()
            
            # Ordena pelo custo 'Real' em ordem decrescente e pega as N principais
            branch_costs = branch_costs.sort_values(by='Real', ascending=False).head(num_top_branches)
            
            # Derrete os dados para plotagem
            if value_type_selection == "Ambos":
                melted_branch_costs = branch_costs.melt(id_vars='Cod. Filial',
                                                        value_vars=['Real', 'Orçado'],
                                                        var_name='Tipo',
                                                        value_name='Valor')
            elif value_type_selection == "Real":
                melted_branch_costs = branch_costs[['Cod. Filial', 'Real']].rename(columns={'Real': 'Valor'})
                melted_branch_costs['Tipo'] = 'Real'
            else: # "Orçado"
                melted_branch_costs = branch_costs[['Cod. Filial', 'Orçado']].rename(columns={'Orçado': 'Valor'})
                melted_branch_costs['Tipo'] = 'Orçado'

            melted_branch_costs['Ano'] = year
            top_branches_data = pd.concat([top_branches_data, melted_branch_costs])
        
        if not top_branches_data.empty:
            # Cria uma coluna combinada 'Tipo_Ano' para coloração distinta
            top_branches_data['Tipo_Ano'] = top_branches_data['Tipo'] + ' ' + top_branches_data['Ano'].astype(str)

            fig_top_branches = px.bar(top_branches_data,
                                        x='Cod. Filial',
                                        y='Valor',
                                        color='Tipo_Ano',
                                        barmode='group',
                                        facet_col='Ano', # Gráficos separados para cada ano
                                        facet_col_wrap=2,
                                        text='Valor',
                                        labels={'Valor': 'Valor (R$)', 'Cod. Filial': 'Código da Filial', 'Tipo_Ano': 'Tipo & Ano'},
                                        height=600,
                                        color_discrete_map=custom_colors) # Usa as mesmas cores personalizadas

            fig_top_branches.update_traces(texttemplate='R$ %{value:,.2f}', textposition='outside')
            fig_top_branches.update_layout(hovermode='x unified',
                                            yaxis_tickprefix='R$ ',
                                            yaxis_tickformat=',.2f',
                                            legend_title_text='')
            fig_top_branches.for_each_annotation(lambda a: a.update(text=a.text.replace("Ano=", "Ano: ")))
            st.plotly_chart(fig_top_branches, use_container_width=True)
        else:
            st.info("Nenhum dado de top filiais encontrado para os filtros selecionados.")
    else:
        st.warning("Selecione pelo menos um ano para visualizar as top filiais com maiores custos.")

    # --- Visão Detalhada dos Dados - Tabela Dinâmica Dinâmica ---
    st.markdown("---")
    st.header("Dados Detalhados (Comparativo Cumulativo por Mês) 📚")
    
    if not filtered_df.empty:
        # Agrupa por Filial, Segmento, Ano, Mês_Num, Nome_Mês e Histórico
        # Calcula a soma de Real e Orçado para cada mês
        monthly_agg = filtered_df.groupby(['Cod. Filial', 'Segmento', 'Ano', 'Mês_Num', 'Nome_Mês', 'Histórico'])[['Real', 'Orçado']].sum().reset_index()
        
        # Ordena para garantir o cálculo cumulativo correto
        monthly_agg = monthly_agg.sort_values(by=['Cod. Filial', 'Segmento', 'Histórico', 'Ano', 'Mês_Num'])
        
        # Calcula as somas cumulativas para Real e Orçado dentro de cada Filial, Segmento, Histórico e Ano
        monthly_agg['Real_Acumulado'] = monthly_agg.groupby(['Cod. Filial', 'Segmento', 'Histórico', 'Ano'])['Real'].cumsum()
        monthly_agg['Orçado_Acumulado'] = monthly_agg.groupby(['Cod. Filial', 'Segmento', 'Histórico', 'Ano'])['Orçado'].cumsum()
        
        # Calcula a variação cumulativa e % variação
        monthly_agg['Variação_Acumulada'] = monthly_agg['Real_Acumulado'] - monthly_agg['Orçado_Acumulado']
        monthly_agg['% Variação_Acumulada'] = (monthly_agg['Variação_Acumulada'] / monthly_agg['Orçado_Acumulado']).replace([np.inf, -np.inf], np.nan).fillna(0) * 100
        
        # Nova estrutura de tabela dinâmica para comparação ano a ano
        # Pivota a tabela para ter os anos como colunas para comparação, incluindo 'Histórico' no índice
        pivot_table_comparison = monthly_agg.pivot_table(
            index=['Cod. Filial', 'Segmento', 'Histórico', 'Nome_Mês', 'Mês_Num'], # Inclui Mês_Num para ordenação
            columns='Ano',
            values=['Real_Acumulado', 'Orçado_Acumulado', 'Variação_Acumulada', '% Variação_Acumulada']
        )

        # Acha os multi-níveis das colunas
        pivot_table_comparison.columns = [f"{col[0].replace('_Acumulado', '')} {col[1]}" for col in pivot_table_comparison.columns]
        
        # Reseta o índice para tornar 'Cod. Filial', 'Segmento', 'Histórico', 'Nome_Mês' colunas normais
        pivot_table_comparison = pivot_table_comparison.reset_index()

        # Ordena por Filial, Segmento, Histórico e Mês_Num para garantir a ordem cronológica dos meses
        pivot_table_comparison = pivot_table_comparison.sort_values(by=['Cod. Filial', 'Segmento', 'Histórico', 'Mês_Num'])

        # Remove Mês_Num, pois não é mais necessário para exibição
        pivot_table_comparison = pivot_table_comparison.drop(columns=['Mês_Num'])

        # Formata colunas numéricas para exibição
        for col in pivot_table_comparison.columns:
            if 'Real' in col or 'Orçado' in col or 'Variação' in col:
                if pd.api.types.is_numeric_dtype(pivot_table_comparison[col]):
                    pivot_table_comparison[col] = pivot_table_comparison[col].fillna(0).apply(lambda x: f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
                    pivot_table_comparison[col] = pivot_table_comparison[col].replace("R$ 0,00", "-")
            elif '% Variação' in col:
                if pd.api.types.is_numeric_dtype(pivot_table_comparison[col]):
                    pivot_table_comparison[col] = pivot_table_comparison[col].fillna(0).apply(lambda x: f"{x:,.2f}%".replace(",", "X").replace(".", ",").replace("X", "."))
                    pivot_table_comparison[col] = pivot_table_comparison[col].replace("0,00%", "-")

        st.dataframe(pivot_table_comparison, use_container_width=True)
    else:
        st.info("Nenhum dado encontrado para os filtros selecionados para a tabela detalhada.")

else:
    st.error("Não foi possível carregar os dados. Verifique o arquivo e o caminho.")
