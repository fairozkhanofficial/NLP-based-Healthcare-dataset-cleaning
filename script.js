// DOM Elements
const fileInput = document.getElementById('fileInput');
const fileChosen = document.getElementById('file-chosen');
const dropArea = document.getElementById('drop-area');
const uploadBtn = document.getElementById('uploadBtn');
const processingSection = document.getElementById('processingSection');
const resultsSection = document.getElementById('resultsSection');
const progressBar = document.getElementById('progressBar');
const progressStatus = document.getElementById('progressStatus');
const downloadBtn = document.getElementById('downloadBtn');
const viewDataBtn = document.getElementById('viewDataBtn');
const dataPreviewModal = document.getElementById('dataPreviewModal');
const dataPreviewTable = document.getElementById('dataPreviewTable');
const closeModal = document.querySelector('.close-modal');

// Stats elements
const cleanedRecordsEl = document.getElementById('cleanedRecords');
const issuesFixedEl = document.getElementById('issuesFixed');
const imputedValuesEl = document.getElementById('imputedValues');
const outliersFixedEl = document.getElementById('outliersFixed');

// Variables to store data
let originalData = null;
let cleanedData = null;
let cleaningStats = null;
let visualizationData = null;
let selectedFile = null;
let charts = {};

// Event Listeners
fileInput.addEventListener('change', handleFileSelect);
uploadBtn.addEventListener('click', processFile);
downloadBtn.addEventListener('click', downloadCleanedData);
viewDataBtn.addEventListener('click', showDataPreview);
closeModal.addEventListener('click', () => dataPreviewModal.style.display = 'none');

// Close modal when clicking outside of it
window.addEventListener('click', (e) => {
    if (e.target === dataPreviewModal) {
        dataPreviewModal.style.display = 'none';
    }
});

// Tab navigation
document.querySelectorAll('.vis-tabs .tab-btn').forEach(button => {
    button.addEventListener('click', () => {
        // Remove active class from all tabs
        document.querySelectorAll('.vis-tabs .tab-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });
        
        // Add active class to clicked tab
        button.classList.add('active');
        document.getElementById(button.dataset.tab).classList.add('active');
    });
});

// Data preview tab navigation
document.querySelectorAll('.tab-buttons .tab-btn').forEach(button => {
    button.addEventListener('click', () => {
        document.querySelectorAll('.tab-buttons .tab-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        button.classList.add('active');
        
        // Display appropriate data
        const previewType = button.dataset.preview;
        if (previewType === 'original') {
            displayTableData(originalData);
        } else {
            displayTableData(cleanedData);
        }
    });
});

// Drag and drop functionality
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropArea.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

['dragenter', 'dragover'].forEach(eventName => {
    dropArea.addEventListener(eventName, highlight, false);
});

['dragleave', 'drop'].forEach(eventName => {
    dropArea.addEventListener(eventName, unhighlight, false);
});

function highlight() {
    dropArea.classList.add('highlight');
}

function unhighlight() {
    dropArea.classList.remove('highlight');
}

dropArea.addEventListener('drop', handleDrop, false);

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    
    if (files.length > 0) {
        handleFiles(files);
    }
}

function handleFiles(files) {
    if (files.length > 0) {
        const file = files[0];
        
        // Check if file is CSV
        if (file.type === 'text/csv' || file.name.endsWith('.csv')) {
            selectedFile = file;
            fileChosen.textContent = file.name;
            uploadBtn.disabled = false;
        } else {
            alert('Please select a CSV file.');
            fileChosen.textContent = 'Choose a CSV file or drop it here';
            uploadBtn.disabled = true;
        }
    }
}

function handleFileSelect(e) {
    const files = e.target.files;
    handleFiles(files);
}

function processFile() {
    if (!selectedFile) {
        alert('Please select a file first.');
        return;
    }

    // Show processing section
    processingSection.classList.remove('hidden');
    uploadBtn.disabled = true;
    
    // Start processing animation
    startProcessingSimulation();
    
    // Read file content
    const reader = new FileReader();
    reader.onload = function(e) {
        // Store original data
        originalData = e.target.result;
        
        // Send data to Python backend
        cleanData(originalData);
    };
    reader.readAsText(selectedFile);
}

async function cleanData(csvData) {
    try {
        // In a real-world scenario, you would send the data to your backend
        // For this demo, we'll simulate the API call
        
        const formData = new FormData();
        formData.append('file', selectedFile);
        
        // Simulate backend processing
        await simulateBackendProcessing();
        
        // In a real implementation, you would use fetch:
        /*
        const response = await fetch('/api/clean_data', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error('Failed to process data');
        }
        
        const result = await response.json();
        cleanedData = result.cleaned_data;
        cleaningStats = result.stats;
        visualizationData = result.visualizations;
        */
        
        // For demo: generate mock cleaned data and stats
        generateMockResults();
        
        // Show results
        showResults();
        
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred while processing the data: ' + error.message);
        resetUI();
    }
}

function startProcessingSimulation() {
    const steps = document.querySelectorAll('.step');
    const totalSteps = steps.length;
    let currentStep = 0;
    
    // Reset all steps
    steps.forEach(step => {
        step.querySelector('i').className = 'fas fa-circle';
    });
    
    // Mark first step as active
    steps[0].querySelector('i').className = 'fas fa-check-circle';
    
    // Initialize progress bar
    progressBar.style.width = '0%';
}

async function simulateBackendProcessing() {
    const steps = document.querySelectorAll('.step');
    const totalSteps = steps.length;
    const statusMessages = [
        'Analyzing dataset structure...',
        'Treating missing values...',
        'Normalizing text fields...',
        'Detecting and handling outliers...',
        'Generating visualizations...'
    ];
    
    for (let i = 0; i < totalSteps; i++) {
        // Update progress bar
        const progress = Math.round((i / (totalSteps - 1)) * 100);
        progressBar.style.width = `${progress}%`;
        
        // Update status message
        progressStatus.textContent = statusMessages[i];
        
        // Mark current step as in progress
        if (i > 0) {
            const prevStep = steps[i-1];
            const currentStep = steps[i];
            
            prevStep.querySelector('i').className = 'fas fa-check-circle';
            currentStep.querySelector('i').className = 'fas fa-spinner fa-spin';
        }
        
        // Wait for step to complete
        await new Promise(resolve => setTimeout(resolve, 1500));
    }
    
    // Mark final step as complete
    steps[totalSteps-1].querySelector('i').className = 'fas fa-check-circle';
    progressBar.style.width = '100%';
    progressStatus.textContent = 'Processing complete!';
    
    await new Promise(resolve => setTimeout(resolve, 1000));
}

function generateMockResults() {
    // Parse original CSV to get structure for mocking
    const rows = originalData.split('\n');
    const headers = rows[0].split(',');
    
    // Generate cleaned data (same structure but "cleaned")
    cleanedData = originalData; // In a real app, this would be different
    
    // Generate cleaning statistics
    cleaningStats = {
        total_records: rows.length - 1, // Minus header row
        cleaned_records: Math.floor((rows.length - 1) * 0.92),
        issues_fixed: Math.floor((rows.length - 1) * 0.45),
        imputed_values: Math.floor((rows.length - 1) * headers.length * 0.08),
        outliers_fixed: Math.floor((rows.length - 1) * 0.12)
    };
    
    // Generate visualization data
    visualizationData = {
        missing_data: {
            labels: ['Before Cleaning', 'After Cleaning'],
            datasets: [{
                label: 'Missing Values (%)',
                data: [8.4, 0.3],
                backgroundColor: ['rgba(231, 76, 60, 0.7)', 'rgba(46, 204, 113, 0.7)']
            }]
        },
        quality_score: {
            labels: ['Data Completeness', 'Format Consistency', 'Value Range', 'Standardization', 'Overall'],
            datasets: [
                {
                    label: 'Before Cleaning',
                    data: [65, 48, 72, 40, 56],
                    backgroundColor: 'rgba(52, 152, 219, 0.5)',
                    borderColor: 'rgba(52, 152, 219, 1)',
                    borderWidth: 1
                },
                {
                    label: 'After Cleaning',
                    data: [98, 95, 97, 92, 96],
                    backgroundColor: 'rgba(46, 204, 113, 0.5)',
                    borderColor: 'rgba(46, 204, 113, 1)',
                    borderWidth: 1
                }
            ]
        },
        numerical_distribution: {
            labels: ['<20', '20-30', '30-40', '40-50', '50-60', '60-70', '70+'],
            datasets: [
                {
                    label: 'Before Cleaning',
                    data: [5, 12, 18, 25, 20, 15, 5],
                    borderColor: 'rgba(52, 152, 219, 1)',
                    backgroundColor: 'rgba(52, 152, 219, 0.2)',
                    tension: 0.4,
                    fill: true
                },
                {
                    label: 'After Cleaning',
                    data: [5, 12, 18, 25, 20, 15, 5],
                    borderColor: 'rgba(46, 204, 113, 1)',
                    backgroundColor: 'rgba(46, 204, 113, 0.2)',
                    tension: 0.4,
                    fill: true
                }
            ]
        },
        categorical_distribution: {
            labels: ['Category A', 'Category B', 'Category C', 'Category D', 'Category E'],
            datasets: [
                {
                    label: 'Before Cleaning',
                    data: [20, 25, 15, 30, 10],
                    backgroundColor: [
                        'rgba(52, 152, 219, 0.7)',
                        'rgba(155, 89, 182, 0.7)',
                        'rgba(52, 73, 94, 0.7)',
                        'rgba(243, 156, 18, 0.7)',
                        'rgba(231, 76, 60, 0.7)'
                    ]
                },
                {
                    label: 'After Cleaning',
                    data: [18, 22, 25, 25, 10],
                    backgroundColor: [
                        'rgba(52, 152, 219, 0.9)',
                        'rgba(155, 89, 182, 0.9)',
                        'rgba(52, 73, 94, 0.9)',
                        'rgba(243, 156, 18, 0.9)',
                        'rgba(231, 76, 60, 0.9)'
                    ]
                }
            ]
        },
        correlation_data: {
            labels: ['Feature A', 'Feature B', 'Feature C', 'Feature D', 'Feature E'],
            datasets: [{
                label: 'Correlation Matrix',
                data: [
                    [1.0, 0.2, -0.3, 0.5, 0.1],
                    [0.2, 1.0, 0.6, 0.3, -0.2],
                    [-0.3, 0.6, 1.0, 0.1, 0.4],
                    [0.5, 0.3, 0.1, 1.0, 0.7],
                    [0.1, -0.2, 0.4, 0.7, 1.0]
                ]
            }]
        },
        term_frequency: {
            labels: ['Hypertension', 'Diabetes', 'Myocardial Infarction', 'COPD', 'Arrhythmia', 'Heart Failure', 'Asthma'],
            datasets: [{
                label: 'Term Frequency',
                data: [42, 38, 15, 22, 18, 12, 25],
                backgroundColor: 'rgba(155, 89, 182, 0.7)',
                borderColor: 'rgba(155, 89, 182, 1)',
                borderWidth: 1
            }]
        },
        text_quality: {
            labels: ['Abbreviation Standardization', 'Spelling Correction', 'Format Consistency', 'Semantic Accuracy'],
            datasets: [
                {
                    label: 'Before Cleaning',
                    data: [45, 62, 38, 56],
                    backgroundColor: 'rgba(52, 152, 219, 0.5)',
                    borderColor: 'rgba(52, 152, 219, 1)',
                    borderWidth: 1
                },
                {
                    label: 'After Cleaning',
                    data: [92, 97, 95, 90],
                    backgroundColor: 'rgba(46, 204, 113, 0.5)',
                    borderColor: 'rgba(46, 204, 113, 1)',
                    borderWidth: 1
                }
            ]
        }
    };
}

function showResults() {
    // Hide processing section
    processingSection.classList.add('hidden');
    
    // Show results section
    resultsSection.classList.remove('hidden');
    
    // Update statistics
    cleanedRecordsEl.textContent = cleaningStats.cleaned_records;
    issuesFixedEl.textContent = cleaningStats.issues_fixed;
    imputedValuesEl.textContent = cleaningStats.imputed_values;
    outliersFixedEl.textContent = cleaningStats.outliers_fixed;
    
    // Create visualizations
    createVisualizations();
}

function createVisualizations() {
    // Missing data comparison chart
    createBarChart('missingDataChart', visualizationData.missing_data);
    
    // Quality score chart
    createRadarChart('qualityScoreChart', visualizationData.quality_score);
    
    // Numerical distribution chart
    createLineChart('numDistChart', visualizationData.numerical_distribution);
    
    // Categorical distribution chart
    createDoughnutChart('catDistChart', visualizationData.categorical_distribution);
    
    // Correlation matrix chart
    createHeatmapChart('correlationChart', visualizationData.correlation_data);
    
    // Term frequency chart
    createBarChart('termFreqChart', visualizationData.term_frequency);
    
    // Text quality chart
    createRadarChart('textQualityChart', visualizationData.text_quality);
}

function createBarChart(canvasId, data) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    
    // Destroy existing chart if it exists
    if (charts[canvasId]) {
        charts[canvasId].destroy();
    }
    
    charts[canvasId] = new Chart(ctx, {
        type: 'bar',
        data: data,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}

function createRadarChart(canvasId, data) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    
    // Destroy existing chart if it exists
    if (charts[canvasId]) {
        charts[canvasId].destroy();
    }
    
    charts[canvasId] = new Chart(ctx, {
        type: 'radar',
        data: data,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    min: 0,
                    max: 100,
                    ticks: {
                        stepSize: 20
                    }
                }
            }
        }
    });
}

function createLineChart(canvasId, data) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    
    // Destroy existing chart if it exists
    if (charts[canvasId]) {
        charts[canvasId].destroy();
    }
    
    charts[canvasId] = new Chart(ctx, {
        type: 'line',
        data: data,
        options: {
            responsive: true,
            maintainAspectRatio: false
        }
    });
}

function createDoughnutChart(canvasId, data) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    
    // Destroy existing chart if it exists
    if (charts[canvasId]) {
        charts[canvasId].destroy();
    }
    
    charts[canvasId] = new Chart(ctx, {
        type: 'doughnut',
        data: data,
        options: {
            responsive: true,
            maintainAspectRatio: false
        }
    });
}

function createHeatmapChart(canvasId, data) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    const labels = data.labels;
    const values = data.datasets[0].data;
    
    // Destroy existing chart if it exists
    if (charts[canvasId]) {
        charts[canvasId].destroy();
    }
    
    // Create a dataset with colors based on correlation values
    const dataset = {
        label: 'Correlation',
        data: [],
        backgroundColor: []
    };
    
    // Flatten the matrix and generate colors
    for (let i = 0; i < values.length; i++) {
        for (let j = 0; j < values[i].length; j++) {
            const value = values[i][j];
            dataset.data.push({
                x: labels[j],
                y: labels[i],
                v: value
            });
            
            // Generate color based on correlation value
            // Red for negative, Blue for positive
            if (value < 0) {
                const intensity = Math.min(Math.abs(value) * 255, 255);
                dataset.backgroundColor.push(`rgba(231, 76, 60, ${Math.abs(value)})`);
            } else {
                const intensity = Math.min(value * 255, 255);
                dataset.backgroundColor.push(`rgba(52, 152, 219, ${value})`);
            }
        }
    }
    
    charts[canvasId] = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [dataset]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    type: 'category',
                    position: 'top',
                    ticks: {
                        display: true
                    }
                },
                y: {
                    type: 'category',
                    position: 'left',
                    reverse: true,
                    ticks: {
                        display: true
                    }
                }
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const point = context.raw;
                            return `${point.x} to ${point.y}: ${point.v.toFixed(2)}`;
                        }
                    }
                },
                legend: {
                    display: false
                }
            }
        }
    });
}

function downloadCleanedData() {
    if (!cleanedData) {
        alert('No cleaned data available');
        return;
    }
    
    // Create a blob with the CSV data
    const blob = new Blob([cleanedData], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    
    // Create a temporary link element
    const a = document.createElement('a');
    a.href = url;
    a.download = 'cleaned_' + selectedFile.name;
    
    // Trigger download
    document.body.appendChild(a);
    a.click();
    
    // Clean up
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

function showDataPreview() {
    if (!originalData || !cleanedData) {
        alert('No data available to preview');
        return;
    }
    
    // Display original data by default
    displayTableData(originalData);
    
    // Show modal
    dataPreviewModal.style.display = 'block';
}

function displayTableData(csvData) {
    // Parse CSV
    const rows = csvData.split('\n');
    if (rows.length <= 1) {
        dataPreviewTable.innerHTML = '<tr><td>No data available</td></tr>';
        return;
    }
    
    const headers = rows[0].split(',');
    
    // Build table HTML
    let tableHTML = '<tr>';
    headers.forEach(header => {
        tableHTML += `<th>${header.trim()}</th>`;
    });
    tableHTML += '</tr>';
    
    // Add data rows (limit to first 100 for performance)
    const rowLimit = Math.min(rows.length, 101);
    for (let i = 1; i < rowLimit; i++) {
        if (rows[i].trim() === '') continue;
        
        const cells = rows[i].split(',');
        tableHTML += '<tr>';
        cells.forEach(cell => {
            tableHTML += `<td>${cell.trim()}</td>`;
        });
        tableHTML += '</tr>';
    }
    
    // Display table
    dataPreviewTable.innerHTML = tableHTML;
}

function resetUI() {
    processingSection.classList.add('hidden');
    resultsSection.classList.add('hidden');
    uploadBtn.disabled = false;
    progressBar.style.width = '0%';
}

// Add window resize handler for charts
window.addEventListener('resize', function() {
    // Redraw all charts
    Object.keys(charts).forEach(id => {
        if (charts[id]) {
            charts[id].resize();
        }
    });
});