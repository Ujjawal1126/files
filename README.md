def extract_text_from_pdf(blob_service_connection_string, form_recognizer_key, form_recognizer_endpoint, text_analytics_key, text_analytics_endpoint, container_name, blob_name):
    # Set up Blob Storage client
    blob_service_client = BlobServiceClient.from_connection_string(blob_service_connection_string)
    container_client = blob_service_client.get_container_client(container_name)
    blob_client = container_client.get_blob_client(blob_name)

    # Download PDF content
    pdf_content = blob_client.download_blob().readall()

    # Set up Form Recognizer client
    form_recognizer_credentials = AzureKeyCredential(form_recognizer_key)
    form_recognizer_client = FormRecognizerClient(form_recognizer_endpoint, form_recognizer_credentials)

    # Extract text from PDF
    poller = form_recognizer_client.begin_recognize_content(pdf_content)
    result = poller.result()

    extracted_text = ""
    for page in result:
        for line in page.lines:
            extracted_text += line.text + " "

    # Set up Text Analytics client
    text_analytics_credentials = AzureKeyCredential(text_analytics_key)
    text_analytics_client = TextAnalyticsClient(text_analytics_endpoint, text_analytics_credentials)

    # Call the Text Analytics API for summarization
    documents = [extracted_text]
    response = text_analytics_client.extract_summary(documents)

    # Extract and print the summary
    for doc in response:
        print("Summary:")
        for sentence in doc['sentences']:
            print(sentence)
        print()
