'''Streamlit client for the Support Ticket Intelligence API.'''

import os

import requests
import streamlit as st


DEFAULT_API_URL = os.getenv('SUPPORT_API_URL', 'http://127.0.0.1:8000')

st.set_page_config(
    page_title='Support Ticket Intelligence',
    page_icon='🎫',
    layout='centered',
)

st.title('🎫 Support Ticket Intelligence')
st.caption('Classify a customer message and route it to the right support queue.')

with st.sidebar:
    st.header('Connection')
    api_url = st.text_input('FastAPI URL', value=DEFAULT_API_URL).strip().rstrip('/')
    if st.button('Check API', use_container_width=True):
        try:
            health_response = requests.get(f'{api_url}/health', timeout=10)
            health_response.raise_for_status()
            health = health_response.json()
            st.success(f'''Ready · device: {health.get('device', 'unknown')}''')
        except (requests.RequestException, ValueError) as exc:
            st.error(f'Could not reach the API: {exc}')

with st.form('ticket_form'):
    ticket = st.text_area(
        'Customer message',
        placeholder='Example: My card was charged twice for the same order.',
        height=160,
        max_chars=10_000,
    )
    submitted = st.form_submit_button(
        'Classify ticket',
        type='primary',
        use_container_width=True,
    )

if submitted:
    if not ticket.strip():
        st.warning('Enter a customer message before classifying it.')
    elif not api_url:
        st.warning('Enter the FastAPI URL in the sidebar.')
    else:
        try:
            with st.spinner('Classifying ticket…'):
                response = requests.post(
                    f'{api_url}/predict',
                    json={'text': ticket},
                    timeout=60,
                )
                response.raise_for_status()
                prediction = response.json()

            confidence = float(prediction['confidence'])
            intent = str(prediction['intent'])
            friendly_intent = intent.replace('_', ' ').title()

            st.subheader('Prediction')
            intent_column, confidence_column = st.columns(2)
            intent_column.metric('Intent', friendly_intent)
            confidence_column.metric('Confidence', f'{confidence:.1%}')
            st.progress(confidence, text=f'Model confidence: {confidence:.1%}')

            if prediction.get('low_confidence'):
                st.warning('Low-confidence prediction — send this ticket for human review.')
            else:
                st.success(f'Suggested routing label: {intent}')
        except requests.HTTPError as exc:
            detail = ''
            try:
                detail = exc.response.json().get('detail', '')
            except (ValueError, AttributeError):
                pass
            st.error(detail or f'The API returned an error: {exc}')
        except (requests.RequestException, KeyError, TypeError, ValueError) as exc:
            st.error(f'Could not classify the ticket: {exc}')
