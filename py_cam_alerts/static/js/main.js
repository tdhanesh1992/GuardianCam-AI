// GuardianCam AI - Frontend JavaScript Controller

document.addEventListener('DOMContentLoaded', () => {
    // UI Elements
    const liveClock = document.getElementById('liveClock');
    const statusPill = document.getElementById('statusPill');
    const statusText = document.getElementById('statusText');
    const fpsVal = document.getElementById('fpsVal');
    const soundToggleBtn = document.getElementById('soundToggleBtn');
    const alertOverlay = document.getElementById('alertOverlay');
    const alertOverlayTitle = document.getElementById('alertOverlayTitle');
    const alertOverlaySub = document.getElementById('alertOverlaySub');

    // Counters
    const valChildren = document.getElementById('valChildren');
    const valFalls = document.getElementById('valFalls');
    const valBreaches = document.getElementById('valBreaches');
    const valHazards = document.getElementById('valHazards');

    // Event Log
    const eventLogList = document.getElementById('eventLogList');

    // State Variables
    let soundEnabled = true;
    let audioCtx = null;
    let knownAlertIds = new Set();
    let chartInstance = null;

    // Live Clock Update
    function updateClock() {
        const now = new Date();
        liveClock.textContent = now.toTimeString().split(' ')[0];
    }
    setInterval(updateClock, 1000);
    updateClock();

    // Web Audio Synthesizer for Safety Alarm Tones
    function initAudio() {
        if (!audioCtx) {
            audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        }
    }

    function playAlarmSound(type = 'danger') {
        if (!soundEnabled) return;
        initAudio();
        if (audioCtx.state === 'suspended') {
            audioCtx.resume();
        }

        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.connect(gain);
        gain.connect(audioCtx.destination);

        const now = audioCtx.currentTime;
        if (type === 'danger') {
            // High-pitched double beep alarm siren
            osc.type = 'sawtooth';
            osc.frequency.setValueAtTime(880, now);
            osc.frequency.exponentialRampToValueAtTime(440, now + 0.25);
            gain.gain.setValueAtTime(0.3, now);
            gain.gain.exponentialRampToValueAtTime(0.01, now + 0.25);
            osc.start(now);
            osc.stop(now + 0.25);
        } else {
            // Subtle notification chime
            osc.type = 'sine';
            osc.frequency.setValueAtTime(587.33, now); // D5 note
            gain.gain.setValueAtTime(0.15, now);
            gain.gain.exponentialRampToValueAtTime(0.01, now + 0.2);
            osc.start(now);
            osc.stop(now + 0.2);
        }
    }

    // Sound Toggle Button
    soundToggleBtn.addEventListener('click', () => {
        soundEnabled = !soundEnabled;
        if (soundEnabled) {
            soundToggleBtn.innerHTML = '<i data-lucide="volume-2"></i> Sound ON';
            soundToggleBtn.classList.remove('btn-outline');
            soundToggleBtn.classList.add('btn-glass');
            playAlarmSound('chime');
        } else {
            soundToggleBtn.innerHTML = '<i data-lucide="volume-x"></i> Muted';
            soundToggleBtn.classList.add('btn-outline');
            soundToggleBtn.classList.remove('btn-glass');
        }
        lucide.createIcons();
    });

    // Chart.js Initialization
    const ctx = document.getElementById('riskChart').getContext('2d');
    chartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Active Children',
                    data: [],
                    borderColor: '#00f2fe',
                    backgroundColor: 'rgba(0, 242, 254, 0.1)',
                    fill: true,
                    tension: 0.4
                },
                {
                    label: 'Falls & Breaches',
                    data: [],
                    borderColor: '#ff1744',
                    backgroundColor: 'rgba(255, 23, 68, 0.1)',
                    fill: true,
                    tension: 0.4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { labels: { color: '#8a99ad' } } },
            scales: {
                x: { ticks: { color: '#8a99ad' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                y: { min: 0, max: 5, ticks: { color: '#8a99ad', stepSize: 1 }, grid: { color: 'rgba(255,255,255,0.05)' } }
            }
        }
    });

    function updateChart(children, incidents) {
        const timeStr = new Date().toTimeString().split(' ')[0];
        const labels = chartInstance.data.labels;
        const ds0 = chartInstance.data.datasets[0].data;
        const ds1 = chartInstance.data.datasets[1].data;

        labels.push(timeStr);
        ds0.push(children);
        ds1.push(incidents);

        if (labels.length > 15) {
            labels.shift();
            ds0.shift();
            ds1.shift();
        }
        chartInstance.update();
    }

    // Polling Stats Endpoint
    async function fetchStats() {
        try {
            const res = await fetch('/api/stats');
            if (!res.ok) return;
            const stats = await res.json();

            // Update FPS & Counters
            fpsVal.textContent = stats.fps;
            valChildren.textContent = stats.children_count;
            valFalls.textContent = stats.fall_events;
            valBreaches.textContent = stats.cradle_breaches;
            valHazards.textContent = stats.hazard_alerts;

            // Update System Status Pill
            statusPill.className = `status-pill ${stats.overall_status.toLowerCase()}`;
            statusText.textContent = `SYSTEM ${stats.overall_status}`;

            // Trigger Alert Overlay & Sound if Danger
            if (stats.overall_status === 'DANGER' && stats.last_alert) {
                alertOverlay.classList.remove('hidden');
                alertOverlayTitle.textContent = stats.last_alert.title || 'DANGER DETECTED!';
                alertOverlaySub.textContent = `Severity: ${stats.last_alert.severity || 'CRITICAL'}`;
                
                if (!knownAlertIds.has(stats.last_alert.id)) {
                    knownAlertIds.add(stats.last_alert.id);
                    playAlarmSound('danger');
                    addEventLog(stats.last_alert);
                }
            } else {
                alertOverlay.classList.add('hidden');
            }

            // Update Chart
            updateChart(stats.children_count, stats.fall_events + stats.cradle_breaches);

        } catch (e) {
            console.error('Error fetching stats:', e);
        }
    }
    setInterval(fetchStats, 900);

    // Event Log Helper
    function addEventLog(alert) {
        const emptyMsg = eventLogList.querySelector('.empty-log-msg');
        if (emptyMsg) emptyMsg.remove();

        const item = document.createElement('div');
        item.className = `event-item ${alert.severity || 'HIGH'}`;
        item.innerHTML = `
            <div>
                <div class="event-title">${alert.title}</div>
                <div class="event-time">${alert.timestamp} | Severity: ${alert.severity}</div>
            </div>
            <i data-lucide="alert-circle" style="color: currentColor"></i>
        `;
        eventLogList.prepend(item);
        lucide.createIcons();
    }

    window.clearEventLog = function() {
        eventLogList.innerHTML = `
            <div class="empty-log-msg">
                <i data-lucide="check-circle-2"></i>
                <p>No safety alerts recorded. Children are safe.</p>
            </div>
        `;
        lucide.createIcons();
        knownAlertIds.clear();
    };

    // Video Source Controls
    window.setDemoSource = async function(path) {
        await fetch('/api/set_source', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ source_type: 'file', source_path: path })
        });
        reloadStream();
    };

    window.setWebcamSource = async function() {
        const idx = document.getElementById('camIndex').value;
        await fetch('/api/set_source', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ source_type: 'webcam', source_path: idx })
        });
        reloadStream();
    };

    window.setUrlSource = async function() {
        const url = document.getElementById('streamUrlInput').value;
        if (!url) return alert('Please enter a valid RTSP or HTTP camera stream URL.');
        await fetch('/api/set_source', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ source_type: 'url', source_path: url })
        });
        reloadStream();
    };

    // File Upload Form
    const videoFileInput = document.getElementById('videoFileInput');
    const uploadForm = document.getElementById('uploadForm');

    uploadForm.addEventListener('click', () => videoFileInput.click());
    videoFileInput.addEventListener('change', async () => {
        if (!videoFileInput.files.length) return;
        const formData = new FormData();
        formData.append('file', videoFileInput.files[0]);

        try {
            const res = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();
            if (data.status === 'success') {
                reloadStream();
            } else {
                alert(data.detail || 'Upload failed');
            }
        } catch (e) {
            alert('Upload error: ' + e.message);
        }
    });

    function reloadStream() {
        const img = document.getElementById('videoStream');
        img.src = '/video_feed?t=' + new Date().getTime();
    }

    // Tab Switcher
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabPanes = document.querySelectorAll('.tab-pane');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            tabPanes.forEach(p => p.classList.remove('active'));

            btn.classList.add('active');
            document.getElementById(`tab-${btn.dataset.tab}`).classList.add('active');
        });
    });

    // Overlay Toggles
    const toggleBoxes = document.getElementById('toggleBoxes');
    const toggleSkeletons = document.getElementById('toggleSkeletons');
    const toggleRoi = document.getElementById('toggleRoi');
    const toggleSpills = document.getElementById('toggleSpills');

    async function sendOverlayToggles() {
        await fetch('/api/toggle_overlays', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                draw_boxes: toggleBoxes.checked,
                draw_skeletons: toggleSkeletons.checked,
                draw_cradle_roi: toggleRoi.checked,
                draw_spills: toggleSpills.checked
            })
        });
    }

    toggleBoxes.addEventListener('change', sendOverlayToggles);
    toggleSkeletons.addEventListener('change', sendOverlayToggles);
    toggleRoi.addEventListener('change', sendOverlayToggles);
    toggleSpills.addEventListener('change', sendOverlayToggles);
});
