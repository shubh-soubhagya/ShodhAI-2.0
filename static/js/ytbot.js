// DOM Elements
const historyBtn = document.getElementById('historyBtn');
const transcriptBtn = document.getElementById('transcriptBtn');
const sidebar = document.getElementById('sidebar');
const newChatBtn = document.getElementById('newChatBtn');
const historyList = document.getElementById('historyList');
const chatMessages = document.getElementById('chatMessages');
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');
const transcriptModal = document.getElementById('transcriptModal');
const transcriptContent = document.getElementById('transcriptContent');
const closeModal = document.querySelector('.close');
const currentChatTitle = document.getElementById('currentChatTitle');

// Chat database and state variables
let chatHistory = JSON.parse(localStorage.getItem('chatHistory')) || [];
let currentChatId = null;
let currentTranscript = null;
let processingVideo = false;
let isLoading = false;

// Initialize the chatbot
function init() {
    renderChatHistory();
    createNewChat();
    setupEventListeners();
}

// Setup event listeners
function setupEventListeners() {
    historyBtn.addEventListener('click', toggleSidebar);
    newChatBtn.addEventListener('click', createNewChat);
    sendBtn.addEventListener('click', handleSendMessage);
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') handleSendMessage();
    });
    transcriptBtn.addEventListener('click', showTranscriptModal);
    closeModal.addEventListener('click', hideTranscriptModal);
    window.addEventListener('click', (e) => {
        if (e.target === transcriptModal) hideTranscriptModal();
    });
}

// Toggle sidebar visibility
function toggleSidebar() {
    sidebar.classList.toggle('active');
}

// Create a new chat
function createNewChat() {
    processingVideo = false;
    currentTranscript = null;
    
    const chatId = Date.now().toString();
    const newChat = {
        id: chatId,
        title: 'New Chat',
        messages: [{
            role: 'bot',
            content: 'Hi! I can extract transcripts from YouTube videos and answer questions about them. Send me a YouTube URL to get started!'
        }],
        transcript: null,
        timestamp: new Date().toISOString()
    };
    
    chatHistory.unshift(newChat);
    saveHistory();
    
    currentChatId = chatId;
    currentChatTitle.textContent = 'New Chat';
    
    renderChatHistory();
    renderCurrentChat();
    
    if (sidebar.classList.contains('active') && window.innerWidth <= 768) {
        toggleSidebar();
    }
}

// Handle sending a message
function handleSendMessage() {
    if (isLoading) return; // Prevent sending while loading
    
    const message = userInput.value.trim();
    if (!message) return;
    
    addMessageToChat('user', message);
    userInput.value = '';
    
    // Check if this is a YouTube URL
    if (isYouTubeUrl(message) && !processingVideo) {
        processingVideo = true;
        handleYouTubeUrl(message);
    } else if (processingVideo) {
        // This is a question about the video
        handleQuestionAboutVideo(message);
    } else {
        // This is not a YouTube URL and we're not processing a video
        addMessageToChat('bot', 'Please send a valid YouTube URL to get started.');
    }
}

// Check if a string is a YouTube URL
function isYouTubeUrl(url) {
    const regex = /^(https?:\/\/)?(www\.)?(youtube\.com|youtu\.?be)\/.+/;
    return regex.test(url);
}

// Handle YouTube URL processing
function handleYouTubeUrl(url) {
    isLoading = true;
    
    // Create loading message
    const loadingId = Date.now().toString();
    addMessageToChat('bot', 'Extracting transcript, please wait... <span class="loading-dots"></span>', loadingId);
    
    // Make API call to extract transcript
    fetch('/ytbot/api/extract-transcript', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ url: url })
    })
    .then(response => response.json())
    .then(data => {
        isLoading = false;
        
        // Remove loading message
        removeMessage(loadingId);
        
        if (data.success) {
            currentTranscript = data.transcript;
            updateChatTitle(url);
            addMessageToChat('bot', 'Transcript extracted successfully! You can now ask me questions about the video.');
            transcriptContent.innerHTML = `<pre>${currentTranscript}</pre>`;
            
            // Save transcript to current chat
            const chatIndex = chatHistory.findIndex(chat => chat.id === currentChatId);
            if (chatIndex !== -1) {
                chatHistory[chatIndex].transcript = currentTranscript;
                saveHistory();
            }
        } else {
            addMessageToChat('bot', `Failed to extract transcript: ${data.error}`);
            processingVideo = false;
        }
    })
    .catch(error => {
        isLoading = false;
        
        // Remove loading message
        removeMessage(loadingId);
        
        console.error('Error:', error);
        addMessageToChat('bot', 'An error occurred while extracting the transcript.');
        processingVideo = false;
    });
}

// Handle question about the current video
function handleQuestionAboutVideo(question) {
    isLoading = true;
    
    // Create loading message
    const loadingId = Date.now().toString();
    addMessageToChat('bot', 'Analyzing your question... <span class="loading-dots"></span>', loadingId);
    
    // Log for debugging
    console.log("Sending question with transcript of length:", 
                currentTranscript ? currentTranscript.length : 0);
    
    // Make API call to ask question
    fetch('/ytbot/api/ask-question', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            question: question,
            transcript: currentTranscript  // Send the transcript with the request
        })
    })
    .then(response => response.json())
    .then(data => {
        isLoading = false;
        
        // Remove loading message
        removeMessage(loadingId);
        
        if (data.success) {
            addMessageToChat('bot', data.answer);
        } else {
            console.error("Error from server:", data.error);
            addMessageToChat('bot', `Sorry, I couldn't answer that: ${data.error}`);
        }
    })
    .catch(error => {
        isLoading = false;
        
        // Remove loading message
        removeMessage(loadingId);
        
        console.error('Error:', error);
        addMessageToChat('bot', 'An error occurred while processing your question.');
    });
}

// Add a message to the current chat
function addMessageToChat(role, content, id = null) {
    // Create and add message element to UI
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    if (id) messageDiv.id = id;
    
    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    
    const messageText = document.createElement('p');
    messageText.innerHTML = content; // Use innerHTML to support loading animation
    
    messageContent.appendChild(messageText);
    messageDiv.appendChild(messageContent);
    chatMessages.appendChild(messageDiv);
    
    // Auto scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    // Save message to current chat history (only if it's not a loading message)
    if (!id) {
        const chatIndex = chatHistory.findIndex(chat => chat.id === currentChatId);
        if (chatIndex !== -1) {
            chatHistory[chatIndex].messages.push({ role, content });
            saveHistory();
        }
    }
}

// Remove a message by ID
function removeMessage(id) {
    const message = document.getElementById(id);
    if (message) {
        message.remove();
    }
}

// Update the title of the current chat
function updateChatTitle(url) {
    let title = 'Chat about ';
    // Extract video ID or use shortened URL
    if (url.includes('youtu.be')) {
        title += url.split('/').pop();
    } else if (url.includes('youtube.com')) {
        const urlParams = new URLSearchParams(url.split('?')[1] || '');
        title += urlParams.get('v') || 'YouTube video';
    } else {
        title += 'YouTube video';
    }
    
    currentChatTitle.textContent = title;
    
    // Update title in chat history
    const chatIndex = chatHistory.findIndex(chat => chat.id === currentChatId);
    if (chatIndex !== -1) {
        chatHistory[chatIndex].title = title;
        saveHistory();
        renderChatHistory();
    }
}

// Render the chat history in the sidebar
function renderChatHistory() {
    historyList.innerHTML = '';
    
    chatHistory.forEach(chat => {
        const historyItem = document.createElement('div');
        historyItem.className = `history-item ${chat.id === currentChatId ? 'active' : ''}`;
        historyItem.dataset.id = chat.id;
        historyItem.textContent = chat.title;
        
        historyItem.addEventListener('click', () => loadChat(chat.id));
        
        historyList.appendChild(historyItem);
    });
}

// Load a chat from history
function loadChat(chatId) {
    const chat = chatHistory.find(c => c.id === chatId);
    if (!chat) return;
    
    currentChatId = chatId;
    currentChatTitle.textContent = chat.title;
    currentTranscript = chat.transcript;
    processingVideo = !!chat.transcript;
    
    if (currentTranscript) {
        transcriptContent.innerHTML = `<pre>${currentTranscript}</pre>`;
    } else {
        transcriptContent.innerHTML = '<p>No transcript available for this chat.</p>';
    }
    
    renderCurrentChat();
    renderChatHistory();
    
    if (sidebar.classList.contains('active') && window.innerWidth <= 768) {
        toggleSidebar();
    }
}

// Render the current chat messages
function renderCurrentChat() {
    chatMessages.innerHTML = '';
    
    const chat = chatHistory.find(c => c.id === currentChatId);
    if (!chat) return;
    
    chat.messages.forEach(msg => {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${msg.role}`;
        
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        
        const messageText = document.createElement('p');
        messageText.innerHTML = msg.content;
        
        messageContent.appendChild(messageText);
        messageDiv.appendChild(messageContent);
        chatMessages.appendChild(messageDiv);
    });
    
    // Auto scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Show transcript modal
function showTranscriptModal() {
    if (!currentTranscript) {
        transcriptContent.innerHTML = '<p>No transcript available yet. Please send a YouTube URL first.</p>';
    }
    transcriptModal.style.display = 'block';
}

// Hide transcript modal
function hideTranscriptModal() {
    transcriptModal.style.display = 'none';
}

// Save chat history to localStorage
function saveHistory() {
    // Limit history to 50 chats to prevent localStorage overflow
    if (chatHistory.length > 50) {
        chatHistory = chatHistory.slice(0, 50);
    }
    localStorage.setItem('chatHistory', JSON.stringify(chatHistory));
}

// Initialize the app
init();