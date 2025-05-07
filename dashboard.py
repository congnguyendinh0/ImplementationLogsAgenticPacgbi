import streamlit as st
import pandas as pd
import sqlite3
import json
import plotly.express as px
import numpy as np
import tempfile
from datetime import datetime
from collections import defaultdict

# Set page configuration
st.set_page_config(
    page_title="Multi-Agent Session Analyzer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Visualization settings
ROLE_COLORS = {
    "CEO": "#FF6B6B",
    "CTO": "#4ECDC4",
    "CMO": "#FFD166",
    "CPO": "#06D6A0",
    "ChatManager": "#A5A5A5",
    "System": "#8884FF",
    "Unknown": "#CCCCCC"
}

def safe_json_loads(json_str):
    """Safely parse JSON strings with error handling"""
    if isinstance(json_str, dict):
        return json_str
    try:
        return json.loads(str(json_str)) if json_str else {}
    except (json.JSONDecodeError, TypeError):
        return {}

def load_database(uploaded_file):
    """Load and preprocess the SQLite database"""
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        
        conn = sqlite3.connect(tmp_path)
        
        # Load all tables
        tables = {
            'agents': pd.read_sql_query('SELECT * FROM agents', conn),
            'chat_completions': pd.read_sql_query('SELECT * FROM chat_completions', conn),
            'events': pd.read_sql_query('SELECT * FROM events', conn),
            'function_calls': pd.read_sql_query('SELECT * FROM function_calls', conn),
            'oai_clients': pd.read_sql_query('SELECT * FROM oai_clients', conn),
            'oai_wrappers': pd.read_sql_query('SELECT * FROM oai_wrappers', conn)
        }
        conn.close()
        
        # Convert timestamps
        for df in tables.values():
            for col in df.columns:
                if 'time' in col.lower() or 'timestamp' in col.lower():
                    df[col] = pd.to_datetime(df[col])
        
        # Parse JSON fields
        tables['agents']['config_dict'] = tables['agents']['init_args'].apply(safe_json_loads)
        tables['chat_completions']['request_dict'] = tables['chat_completions']['request'].apply(safe_json_loads)
        tables['chat_completions']['response_dict'] = tables['chat_completions']['response'].apply(safe_json_loads)
        tables['events']['state_dict'] = tables['events']['json_state'].apply(safe_json_loads)
        tables['function_calls']['args_dict'] = tables['function_calls']['args'].apply(safe_json_loads)
        tables['function_calls']['returns_dict'] = tables['function_calls']['returns'].apply(safe_json_loads)
        tables['oai_clients']['config_dict'] = tables['oai_clients']['init_args'].apply(safe_json_loads)
        tables['oai_wrappers']['config_dict'] = tables['oai_wrappers']['init_args'].apply(safe_json_loads)
        
        # Enrich chat completions
        cc = tables['chat_completions']
        if not cc.empty:
            cc['duration'] = (cc['end_time'] - cc['start_time']).dt.total_seconds()
            cc['date'] = cc['start_time'].dt.date
            cc['hour'] = cc['start_time'].dt.hour
            cc['day_of_week'] = cc['start_time'].dt.day_name()
            cc['request_size'] = cc['request'].apply(lambda x: len(str(x)))
            cc['response_size'] = cc['response'].apply(lambda x: len(str(x)))
            if 'source_name' in cc.columns:
                cc['source_name'] = cc['source_name'].str.upper().str.strip()
            
            # Token extraction
            def extract_tokens(response_dict):
                usage = response_dict.get('usage', {})
                return (
                    int(usage.get('total_tokens', 0)),
                    int(usage.get('prompt_tokens', 0)),
                    int(usage.get('completion_tokens', 0))
                )
            
            cc[['total_tokens', 'prompt_tokens', 'completion_tokens']] = cc['response_dict'].apply(
                lambda x: pd.Series(extract_tokens(x))
            )
            
            cc['cost_per_token'] = np.where(
                cc['total_tokens'] > 0,
                cc['cost'] / cc['total_tokens'],
                0
            )
            cc['completion_ratio'] = np.where(
                cc['total_tokens'] > 0,
                cc['completion_tokens'] / cc['total_tokens'],
                0
            )
        
        return tables
    
    except Exception as e:
        st.error(f"Error loading database: {str(e)}")
        return None
    finally:
        if 'tmp_path' in locals():
            import os
            try:
                os.unlink(tmp_path)
            except:
                pass

def build_unified_session_view(tables):
    """Create a unified chronological view of all session events"""
    unified_events = []
    id_to_session = {}

    # Build ID to Session Mapping
    if 'agents' in tables and not tables['agents'].empty:
        for _, row in tables['agents'].iterrows():
            if 'session_id' in row and pd.notna(row['session_id']):
                session_id = row['session_id']
                if 'wrapper_id' in row and pd.notna(row['wrapper_id']):
                    id_to_session[row['wrapper_id']] = session_id
                if 'agent_id' in row and pd.notna(row['agent_id']):
                    id_to_session[row['agent_id']] = session_id

    if 'oai_clients' in tables and not tables['oai_clients'].empty:
        for _, row in tables['oai_clients'].iterrows():
            if 'session_id' in row and pd.notna(row['session_id']):
                session_id = row['session_id']
                if 'client_id' in row and pd.notna(row['client_id']):
                    id_to_session[row['client_id']] = session_id
                if 'wrapper_id' in row and pd.notna(row['wrapper_id']):
                    id_to_session[row['wrapper_id']] = session_id

    if 'oai_wrappers' in tables and not tables['oai_wrappers'].empty:
        for _, row in tables['oai_wrappers'].iterrows():
            if 'session_id' in row and pd.notna(row['session_id']):
                session_id = row['session_id']
                if 'wrapper_id' in row and pd.notna(row['wrapper_id']):
                    id_to_session[row['wrapper_id']] = session_id

    # Process DataFrames into Unified List
    if 'agents' in tables and not tables['agents'].empty:
        for _, row in tables['agents'].iterrows():
            if 'timestamp' in row and 'session_id' in row and 'wrapper_id' in row:
                unified_events.append({
                    'timestamp': row['timestamp'],
                    'session_id': row['session_id'],
                    'type': 'agent_config',
                    'source_name': row.get('source_name', 'Agent'),
                    'source_id': row['wrapper_id'],
                    'details': {
                        'config': row.get('config_dict', {}),
                        'class': row.get('agent_class_name', 'Unknown'),
                        'agent_id': row.get('agent_id')
                    }
                })

    if 'events' in tables and not tables['events'].empty:
        for _, row in tables['events'].iterrows():
            if 'timestamp' in row and 'session_id' in row and 'source_id' in row:
                event_type = 'event_received_message' if row.get('event_name') == 'received_message' else 'event_other'
                unified_events.append({
                    'timestamp': row['timestamp'],
                    'session_id': row['session_id'],
                    'type': event_type,
                    'source_name': row.get('source_name', 'Unknown'),
                    'source_id': row['source_id'],
                    'details': {
                        'event_name': row.get('event_name', 'Unknown'),
                        'data': row.get('state_dict', {})
                    }
                })

    if 'chat_completions' in tables and not tables['chat_completions'].empty:
        for _, row in tables['chat_completions'].iterrows():
            if 'start_time' in row and 'session_id' in row and 'client_id' in row and 'invocation_id' in row:
                unified_events.append({
                    'timestamp': row['start_time'],
                    'session_id': row['session_id'],
                    'type': 'llm_call_start',
                    'source_name': row.get('source_name', 'Unknown'),
                    'source_id': row['client_id'],
                    'invocation_id': row['invocation_id'],
                    'details': {
                        'request': row.get('request_dict', {}),
                        'model': row.get('request_dict', {}).get('model', 'Unknown'),
                        'wrapper_id': row.get('wrapper_id')
                    }
                })
                unified_events.append({
                    'timestamp': row['end_time'],
                    'session_id': row['session_id'],
                    'type': 'llm_call_end',
                    'source_name': row.get('source_name', 'Unknown'),
                    'source_id': row['client_id'],
                    'invocation_id': row['invocation_id'],
                    'details': {
                        'response': row.get('response_dict', {}),
                        'cost': row.get('cost', 0),
                        'latency': row.get('latency', 0),
                        'is_cached': row.get('is_cached', False),
                        'wrapper_id': row.get('wrapper_id')
                    }
                })

    if 'function_calls' in tables and not tables['function_calls'].empty:
        for _, row in tables['function_calls'].iterrows():
            source_id = row.get('source_id')
            session_id = id_to_session.get(source_id) if source_id else None

            if session_id and 'timestamp' in row:
                unified_events.append({
                    'timestamp': row['timestamp'],
                    'session_id': session_id,
                    'type': 'function_call',
                    'source_name': row.get('source_name', 'Unknown'),
                    'source_id': source_id,
                    'details': {
                        'function_name': row.get('function_name', 'Unknown'),
                        'args': row.get('args_dict', {}),
                        'returns': row.get('returns_dict', {})
                    }
                })

    if 'oai_clients' in tables and not tables['oai_clients'].empty:
        for _, row in tables['oai_clients'].iterrows():
            if 'timestamp' in row and 'session_id' in row and 'client_id' in row:
                unified_events.append({
                    'timestamp': row['timestamp'],
                    'session_id': row['session_id'],
                    'type': 'client_config',
                    'source_name': 'System',
                    'source_id': row['client_id'],
                    'details': {
                        'config': row.get('config_dict', {}),
                        'wrapper_id': row.get('wrapper_id')
                    }
                })

    if 'oai_wrappers' in tables and not tables['oai_wrappers'].empty:
        for _, row in tables['oai_wrappers'].iterrows():
            if 'timestamp' in row and 'session_id' in row and 'wrapper_id' in row:
                unified_events.append({
                    'timestamp': row['timestamp'],
                    'session_id': row['session_id'],
                    'type': 'wrapper_config',
                    'source_name': 'System',
                    'source_id': row['wrapper_id'],
                    'details': {
                        'config': row.get('config_dict', {})
                    }
                })

    # Group by session and sort chronologically
    session_data = defaultdict(list)
    valid_events = [e for e in unified_events if pd.notna(e.get('timestamp'))]
    for event in sorted(valid_events, key=lambda x: x['timestamp']):
        if pd.notna(event.get('session_id')):
            session_data[event['session_id']].append(event)
    
    return session_data

def calculate_session_metrics(session_id, session_events, chat_completions_df):
    """Calculate summary metrics for a session"""
    if not session_events:
        return {}
    
    # Basic timing
    start_time = min(e['timestamp'] for e in session_events)
    end_time = max(e['timestamp'] for e in session_events)
    duration = (end_time - start_time).total_seconds()
    
    # Count different event types
    event_counts = defaultdict(int)
    agent_names = set()
    function_names = set()
    llm_models = set()
    status = 'Completed'
    
    for event in session_events:
        event_counts[event['type']] += 1
        agent_names.add(event['source_name'])
        
        if event['type'] == 'function_call':
            function_names.add(event['details']['function_name'])
        elif event['type'] == 'llm_call_start':
            llm_models.add(event['details']['model'])
        elif event['type'] == 'event_received_message':
            if 'exitcode' in str(event['details']['data']):
                status = 'Failed'
    
    # Get cost/token metrics from chat completions
    session_cc = chat_completions_df[chat_completions_df['session_id'] == session_id]
    total_cost = session_cc['cost'].sum()
    total_tokens = session_cc['total_tokens'].sum()
    
    return {
        'session_id': session_id,
        'start_time': start_time,
        'end_time': end_time,
        'duration': duration,
        'status': status,
        'num_messages': event_counts.get('event_received_message', 0),
        'num_llm_calls': event_counts.get('llm_call_start', 0),
        'num_function_calls': event_counts.get('function_call', 0),
        'agents': list(agent_names),
        'functions': list(function_names),
        'models': list(llm_models),
        'total_cost': total_cost,
        'total_tokens': total_tokens
    }

def display_session_metrics(metrics):
    """Display summary metrics for a session"""
    st.header("📊 Session Metrics")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Status", metrics['status'])
        st.metric("Start Time", metrics['start_time'].strftime('%Y-%m-%d %H:%M:%S'))
    with col2:
        st.metric("Duration", f"{metrics['duration']:.1f} sec")
        st.metric("End Time", metrics['end_time'].strftime('%Y-%m-%d %H:%M:%S'))
    with col3:
        st.metric("Total Cost", f"${metrics['total_cost']:.4f}")
        st.metric("Total Tokens", f"{metrics['total_tokens']:,}")
    
    st.subheader("Participants")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("**Agents:**")
        for agent in metrics['agents']:
            st.write(f"- {agent}")
    with col2:
        st.write("**Functions Used:**")
        for func in metrics['functions']:
            st.write(f"- {func}")
    with col3:
        st.write("**LLM Models:**")
        for model in metrics['models']:
            st.write(f"- {model}")

def enhanced_render_event(event):
    """Enhanced event rendering with role-based styling"""
    timestamp = event['timestamp'].strftime('%H:%M:%S.%f')[:-3]
    source = event['source_name']
    role = source.split('_')[-1] if '_' in source else source
    details = event['details']
    color = ROLE_COLORS.get(role, ROLE_COLORS["Unknown"])
    
    if event['type'] == 'event_received_message':
        data = details.get('data', {})
        message = data.get('message', {})
        content = message.get('content', '')
        sender = message.get('name', 'Unknown')
        
        if "chat_manager" in source.lower() or "chatmanager" in source.lower():
            st.markdown(
                f"""<div style="border-left: 4px solid {ROLE_COLORS['ChatManager']}; 
                    padding-left: 10px; background-color: #F8F9FA; 
                    padding: 10px; border-radius: 5px; margin: 5px 0;">
                    <strong>🔄 {timestamp} [Chat Manager]</strong><br>
                    {content}
                </div>""", 
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""<div style="border-left: 4px solid {color}; 
                    padding-left: 10px; background-color: #FFFFFF; 
                    padding: 10px; border-radius: 5px; margin: 5px 0;">
                    <strong><span style="color: {color}">💬 {timestamp} {sender}</span> → {source}</strong><br>
                    {content}
                </div>""", 
                unsafe_allow_html=True
            )
    
    elif event['type'] in ('llm_call_start', 'llm_call_end'):
        arrow = "→" if event['type'] == 'llm_call_start' else "←"
        st.markdown(
            f"""<div style="border-left: 4px solid {ROLE_COLORS['System']}; 
                padding-left: 10px; background-color: #F0F4FF; 
                padding: 10px; border-radius: 5px; margin: 5px 0;">
                <strong>🧠 {timestamp} {source} {arrow} LLM ({details.get('model', 'Unknown')})</strong>
            </div>""", 
            unsafe_allow_html=True
        )
        with st.expander("Details", expanded=False):
            st.json(details)
    
    elif event['type'] == 'function_call':
        st.markdown(
            f"""<div style="border-left: 4px solid {color}; 
                padding-left: 10px; background-color: #FFF4F4; 
                padding: 10px; border-radius: 5px; margin: 5px 0;">
                <strong>🔧 {timestamp} {source} → {details['function_name']}</strong>
            </div>""", 
            unsafe_allow_html=True
        )
        with st.expander("Details", expanded=False):
            st.write("**Arguments:**")
            st.json(details['args'])
            st.write("**Returns:**")
            if isinstance(details['returns'], dict) and 'error' in details['returns']:
                st.error(details['returns'])
            else:
                st.json(details['returns'])
    
    elif event['type'] in ('agent_config', 'client_config', 'wrapper_config'):
        st.markdown(
            f"""<div style="border-left: 4px solid {ROLE_COLORS['System']}; 
                padding-left: 10px; background-color: #F5F5F5; 
                padding: 10px; border-radius: 5px; margin: 5px 0;">
                <strong>⚙️ {timestamp} {source} - Configuration</strong>
            </div>""", 
            unsafe_allow_html=True
        )
        with st.expander("Details", expanded=False):
            st.json(details['config'])

def enhanced_display_session_trace(session_events):
    """Display trace using enhanced rendering"""
    st.header("📜 Enhanced Session Trace")
    
    for event in session_events:
        enhanced_render_event(event)

def display_session_trace(session_events):
    """Display the full chronological trace of a session"""
    st.header("📜 Session Trace")
    
    for event in session_events:
        timestamp = event['timestamp'].strftime('%H:%M:%S.%f')[:-3]
        source = event['source_name']
        details = event['details']
        
        if event['type'] == 'event_received_message':
            content = details['data'].get('content', '')
            sender = details['data'].get('name', 'Unknown')
            
            if 'exitcode' in str(content):
                with st.expander(f"❌ {timestamp} {source} - Error Message", expanded=True):
                    st.error(content)
            else:
                with st.expander(f"💬 {timestamp} {sender} → {source}", expanded=False):
                    if '```' in content:
                        parts = content.split('```')
                        for i, part in enumerate(parts):
                            if i % 2 == 1:  # Code block
                                st.code(part, language='python')
                            else:
                                st.markdown(part)
                    else:
                        st.markdown(content)
        
        elif event['type'] == 'llm_call_start':
            with st.expander(f"🧠 {timestamp} {source} → {details['model']}", expanded=False):
                st.json(details['request'])
        
        elif event['type'] == 'llm_call_end':
            with st.expander(f"🧠 {timestamp} {source} ← Response ({details.get('latency', 0):.2f}s, ${details.get('cost', 0):.4f})", expanded=False):
                st.json(details['response'])
        
        elif event['type'] == 'function_call':
            with st.expander(f"🔧 {timestamp} {source} → {details['function_name']}", expanded=False):
                st.write("**Arguments:**")
                st.json(details['args'])
                st.write("**Returns:**")
                if isinstance(details['returns'], dict) and 'error' in details['returns']:
                    st.error(details['returns'])
                else:
                    st.json(details['returns'])
        
        elif event['type'] in ('agent_config', 'client_config', 'wrapper_config'):
            with st.expander(f"⚙️ {timestamp} {source} - Configuration", expanded=False):
                st.json(details['config'])

def visualize_conversation_flow(session_events):
    """Enhanced visualization of the conversation flow"""
    st.header("💬 Conversation Flow")
    
    # Extract messages from both event_received_message and chat completions
    message_events = []
    
    for event in session_events:
        # Handle event_received_message
        if event['type'] == 'event_received_message':
            data = event['details'].get('data', {})
            if isinstance(data, dict):
                # Handle both direct message content and nested message structure
                content = data.get('content', '')
                if not content and 'message' in data:
                    message = data['message']
                    if isinstance(message, dict):
                        content = message.get('content', '')
                
                if content:
                    if len(content) > 300:
                        content = content[:297] + "..."
                    
                    message_events.append({
                        'timestamp': event['timestamp'],
                        'sender': data.get('name', event['source_name']),
                        'receiver': event['source_name'],
                        'role': data.get('name', event['source_name']).split('_')[-1] if '_' in data.get('name', event['source_name']) else data.get('name', event['source_name']),
                        'content': content,
                        'full_content': str(content),
                        'type': 'message',
                        'source_name': event['source_name']
                    })
        
        # Handle llm_call_start
        elif event['type'] == 'llm_call_start':
            request = event['details'].get('request', {})
            messages = request.get('messages', [])
            for msg in messages:
                if isinstance(msg, dict):
                    content = str(msg.get('content', ''))
                    if len(content) > 300:
                        content = content[:297] + "..."
                    
                    message_events.append({
                        'timestamp': event['timestamp'],
                        'sender': msg.get('role', 'System'),
                        'receiver': event['source_name'],
                        'role': msg.get('role', 'System'),
                        'content': content,
                        'full_content': str(msg.get('content', '')),
                        'type': 'message',
                        'source_name': event['source_name']
                    })
        
        # Handle llm_call_end
        elif event['type'] == 'llm_call_end':
            response = event['details'].get('response', {})
            if isinstance(response, dict):
                choices = response.get('choices', [])
                for choice in choices:
                    if isinstance(choice, dict):
                        message = choice.get('message', {})
                        if message:
                            content = str(message.get('content', ''))
                            if len(content) > 300:
                                content = content[:297] + "..."
                            
                            message_events.append({
                                'timestamp': event['timestamp'],
                                'sender': event['source_name'],
                                'receiver': 'System',
                                'role': event['source_name'].split('_')[-1] if '_' in event['source_name'] else event['source_name'],
                                'content': content,
                                'full_content': str(message.get('content', '')),
                                'type': 'message',
                                'source_name': event['source_name']
                            })
        
        # Handle function_call
        elif event['type'] == 'function_call':
            message_events.append({
                'timestamp': event['timestamp'],
                'sender': event['source_name'],
                'receiver': 'System',
                'role': event['source_name'].split('_')[-1] if '_' in event['source_name'] else event['source_name'],
                'content': f"Function call: {event['details'].get('function_name', 'Unknown')}",
                'full_content': str(event['details']),
                'type': 'function',
                'source_name': event['source_name']
            })
    
    if not message_events:
        st.warning("No conversation messages found in this session")
        return
    
    # Create conversation dataframe
    conv_df = pd.DataFrame(message_events)
    
    # Debug information
    st.write(f"Number of messages: {len(message_events)}")
    st.write("Message types:", conv_df['type'].value_counts().to_dict())
    st.write("Roles:", conv_df['role'].value_counts().to_dict())
    
    # Create two columns for the layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Chat-like visualization
        st.subheader("Chat View")
        
        # Add custom CSS for better styling
        st.markdown("""
            <style>
            .chat-container {
                height: 600px;
                overflow-y: auto;
                padding: 15px;
                background-color: #f8f9fa;
                border-radius: 10px;
                border: 1px solid #dee2e6;
            }
            .message-bubble {
                margin: 8px 0;
                padding: 8px 12px;
                border-radius: 12px;
                max-width: 85%;
                font-size: 0.85em;
                line-height: 1.3;
                box-shadow: 0 1px 2px rgba(0,0,0,0.1);
            }
            .message-header {
                font-size: 0.8em;
                font-weight: 600;
                margin-bottom: 3px;
            }
            .message-content {
                font-size: 0.85em;
                line-height: 1.3;
                white-space: pre-wrap;
            }
            .message-time {
                font-size: 0.7em;
                color: #666;
                margin-top: 3px;
                text-align: right;
            }
            .expand-button {
                font-size: 0.75em;
                color: #666;
                cursor: pointer;
                text-decoration: underline;
                margin-top: 3px;
            }
            .code-block {
                background-color: #f8f9fa;
                padding: 8px;
                border-radius: 4px;
                font-family: monospace;
                font-size: 0.8em;
                margin: 4px 0;
                overflow-x: auto;
            }
            </style>
        """, unsafe_allow_html=True)
        
        # Create the chat container
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # Group messages by timestamp to show them in order
        for _, row in conv_df.sort_values('timestamp').iterrows():
            # Determine message alignment and styling based on source
            source_name = row['source_name'].lower()
            is_system = source_name in ['system', 'initializer']
            is_ceo = 'ceo' in source_name
            is_manager = 'chat_manager' in source_name
            is_cmo = 'cmo' in source_name
            is_cpo = 'cpo' in source_name
            is_cto = 'cto' in source_name
            
            # Define colors for different roles
            role_colors = {
                'ceo': '#FF6B6B',
                'chat_manager': '#A5A5A5',
                'cmo': '#FFD166',
                'cpo': '#06D6A0',
                'cto': '#4ECDC4',
                'system': '#8884FF',
                'initializer': '#8884FF'
            }
            
            # Get the appropriate color
            color = role_colors.get(source_name, '#CCCCCC')
            
            # Process content for code blocks
            content = row['content']
            if '```' in content:
                parts = content.split('```')
                processed_content = ''
                for i, part in enumerate(parts):
                    if i % 2 == 1:  # Code block
                        processed_content += f'<div class="code-block">{part}</div>'
                    else:
                        processed_content += part
            else:
                processed_content = content
            
            # Create message container with appropriate styling
            message_style = f"""
                <div class="message-bubble" style="
                    {'margin-left: auto;' if not is_system else 'margin-right: auto;'}
                    background-color: {'#FFFFFF' if not is_system else '#F5F5F5'};
                    border-left: 3px solid {color};
                ">
                    <div class="message-header" style="color: {color};">
                        {row['sender']} ({row['role']})
                    </div>
                    <div class="message-content">
                        {processed_content}
                    </div>
                    <div class="message-time">
                        {row['timestamp'].strftime('%H:%M:%S')}
                    </div>
                    {f'<div class="expand-button" onclick="toggleContent(this)">Show more</div>' if len(row['full_content']) > 300 else ''}
                </div>
            """
            st.markdown(message_style, unsafe_allow_html=True)
        
        # Add JavaScript for expand/collapse functionality
        st.markdown("""
            <script>
            function toggleContent(element) {
                const bubble = element.parentElement;
                const content = bubble.querySelector('.message-content');
                const fullContent = bubble.dataset.fullContent;
                
                if (element.textContent === 'Show more') {
                    content.innerHTML = fullContent;
                    element.textContent = 'Show less';
                } else {
                    content.innerHTML = fullContent.substring(0, 297) + '...';
                    element.textContent = 'Show more';
                }
            }
            </script>
        """, unsafe_allow_html=True)
        
        # Close the chat container
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Sidebar with statistics and filters
        st.subheader("Session Statistics")
        
        # Message counts by role
        role_counts = conv_df['role'].value_counts()
        st.write("Messages by Role:")
        for role, count in role_counts.items():
            st.metric(role, count)
        
        # Time span
        time_span = conv_df['timestamp'].max() - conv_df['timestamp'].min()
        st.metric("Conversation Duration", f"{time_span.total_seconds():.1f}s")
        
        # Add filters
        st.subheader("Filters")
        selected_roles = st.multiselect(
            "Show messages from:",
            options=conv_df['role'].unique(),
            default=conv_df['role'].unique()
        )
        
        # Add a search box
        search_term = st.text_input("Search in messages:", "")
        
        # Add a clear button
        if st.button("Clear Filters"):
            st.experimental_rerun()

def display_analytics_view(tables):
    """Display the aggregated analytics view"""
    st.header("📈 Session Analytics")
    
    # Get chat completions data
    cc = tables['chat_completions']
    
    # Overview metrics
    st.subheader("Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Sessions", cc['session_id'].nunique())
    with col2:
        st.metric("Total LLM Calls", len(cc))
    with col3:
        st.metric("Total Cost", f"${cc['cost'].sum():.2f}")
    
    # Cost analysis
    st.subheader("Cost Analysis")
    tab1, tab2 = st.tabs(["By Session", "Over Time"])
    
    with tab1:
        session_costs = cc.groupby('session_id')['cost'].sum().reset_index()
        fig1 = px.bar(
            session_costs.sort_values('cost', ascending=False).head(20),
            x='session_id',
            y='cost',
            title='Top 20 Sessions by Cost',
            labels={'cost': 'Total Cost ($)', 'session_id': 'Session ID'}
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with tab2:
        daily_costs = cc.groupby('date')['cost'].sum().reset_index()
        fig2 = px.line(
            daily_costs,
            x='date',
            y='cost',
            title='Daily Cost Trend',
            labels={'cost': 'Total Cost ($)', 'date': 'Date'}
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Token analysis
    st.subheader("Token Usage")
    tab1, tab2 = st.tabs(["By Model", "Over Time"])
    
    with tab1:
        model_tokens = cc.groupby(cc['request_dict'].apply(lambda x: x.get('model', 'Unknown')))['total_tokens'].sum().reset_index()
        fig3 = px.bar(
            model_tokens,
            x='request_dict',
            y='total_tokens',
            title='Total Tokens by Model',
            labels={'total_tokens': 'Total Tokens', 'request_dict': 'Model'}
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    with tab2:
        cc['hour'] = cc['start_time'].dt.hour
        hourly_tokens = cc.groupby('hour')['total_tokens'].sum().reset_index()
        fig4 = px.bar(
            hourly_tokens,
            x='hour',
            y='total_tokens',
            title='Hourly Token Usage',
            labels={'total_tokens': 'Total Tokens', 'hour': 'Hour of Day'}
        )
        st.plotly_chart(fig4, use_container_width=True)

def main():
    """Main application function"""
    st.title("Multi-Agent Session Analyzer")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "📂 Upload SQLite Database",
        type=["db", "sqlite"],
        help="Upload your multi-agent session database file"
    )
    
    if uploaded_file is not None:
        # Load and process data
        tables = load_database(uploaded_file)
        if tables is None:
            return
        
        # Build unified session view
        session_data = build_unified_session_view(tables)
        
        # Calculate session metrics
        all_session_metrics = [
            calculate_session_metrics(sid, events, tables['chat_completions'])
            for sid, events in session_data.items()
        ]
        metrics_df = pd.DataFrame(all_session_metrics)
        
        # Add toggle for enhanced view
        enhanced_view = st.sidebar.checkbox("Enhanced Conversation View", value=True)
        
        # Navigation
        view_mode = st.sidebar.radio(
            "View Mode",
            ["Session Trace", "Analytics Dashboard"],
            key="view_mode_selector"
        )
        
        if view_mode == "Session Trace":
            # Session selection
            if not metrics_df.empty:
                selected_session = st.sidebar.selectbox(
                    "Select Session",
                    options=metrics_df.sort_values('start_time', ascending=False)['session_id'],
                    format_func=lambda x: f"{x} ({metrics_df[metrics_df['session_id']==x]['status'].iloc[0]})",
                    key="session_selector"
                )
                
                # Display selected session
                session_metrics = metrics_df[metrics_df['session_id'] == selected_session].iloc[0]
                
                # Add tabs for different views
                tab1, tab2 = st.tabs(["Conversation Flow", "Detailed Trace"])
                
                with tab1:
                    visualize_conversation_flow(session_data[selected_session])
                
                with tab2:
                    display_session_metrics(session_metrics)
                    if enhanced_view:
                        st.info("Enhanced view enabled - showing formatted conversation")
                        enhanced_display_session_trace(session_data[selected_session])
                    else:
                        display_session_trace(session_data[selected_session])
            else:
                st.warning("No valid sessions found in the database")
        
        else:  # Analytics Dashboard
            display_analytics_view(tables)
    
    else:
        st.info("Please upload a SQLite database file to begin analysis")

if __name__ == "__main__":
    main()