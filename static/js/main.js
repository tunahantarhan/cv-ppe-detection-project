document.addEventListener("DOMContentLoaded", () => {
    // Karanlık / Açık Tema Switch Mantığı
    const themeToggle = document.getElementById("theme-toggle");
    if (!themeToggle) return;
    
    const currentTheme = localStorage.getItem("theme") || "light";
    document.documentElement.setAttribute("data-theme", currentTheme);
    
    // SVG İkonları
    const sunIcon = `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="5"></circle><line x1="12" y1="1" x2="12" y2="3"></line><line x1="12" y1="21" x2="12" y2="23"></line><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line><line x1="1" y1="12" x2="3" y2="12"></line><line x1="21" y1="12" x2="23" y2="12"></line><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line></svg>`;
    const moonIcon = `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path></svg>`;

    themeToggle.innerHTML = currentTheme === "dark" ? sunIcon : moonIcon;

    themeToggle.addEventListener("click", () => {
        let theme = document.documentElement.getAttribute("data-theme");
        let newTheme = theme === "dark" ? "light" : "dark";
        document.documentElement.setAttribute("data-theme", newTheme);
        localStorage.setItem("theme", newTheme);
        themeToggle.innerHTML = newTheme === "dark" ? sunIcon : moonIcon;
    });
});

// Kamera Açma - Kapama Mantığı
function toggleCamera() {
    const feed = document.getElementById("camera-feed");
    const btn = document.getElementById("cam-btn");
    if (!feed || !btn) return;

    if (feed.src.includes("video_feed")) {
        feed.src = "";
        feed.style.display = "none";
        btn.textContent = "Kamerayı Başlat";
        btn.className = "btn";
    } else {
        feed.src = "/video_feed";
        feed.style.display = "block";
        btn.textContent = "Kamerayı Durdur";
        btn.className = "btn btn-danger";
    }
}

// Lokalden Video Yükleme Mantığı
async function uploadAndPlay() {
    const fileInput = document.getElementById("video-file");
    if (!fileInput || !fileInput.files.length) {
        alert("Lütfen analize sokmak için bir video dosyası seçin.");
        return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    const btn = document.querySelector("#upload-controls button");
    const originalText = btn.textContent;
    btn.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12a9 9 0 1 1-6.219-8.56"></path></svg> YÜKLENİYOR...`;    btn.disabled = true;

    try {
        const response = await fetch("/upload", { method: "POST", body: formData });
        if (response.ok) {
            const data = await response.json();
            startVideo("upload", data.filename);

            const pastList = document.getElementById("past-uploads-list");
            if (pastList) {
                if (pastList.innerHTML.includes("Henüz yüklenmiş bir video bulunmuyor")) {
                    pastList.innerHTML = "";
                }
                if (!pastList.innerHTML.includes(data.filename)) {
                    const newBtn = document.createElement("button");
                    newBtn.className = "btn";
                    newBtn.style = "font-size: 12px; padding: 8px 12px;";
                    newBtn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"></path></svg> ${data.filename}`;
                    newBtn.onclick = () => startVideo("upload", data.filename);
                    pastList.appendChild(newBtn);
                }
            }
        } else {
            alert("Yükleme başarısız oldu. Dosya boyutu 50 MB'ı aşmış olabilir.");
        }
    } catch (error) {
        console.error(error);
        alert("Sunucu ile iletişimde bir hata oluştu.");
    } finally {
        btn.textContent = originalText;
        btn.disabled = false;
    }
}

// Video Oynatma Kontrolleri
function startVideo(type, filename) {
    const uploadControls = document.getElementById("upload-controls");
    const playerArea = document.getElementById("video-player-area");
    const feed = document.getElementById("test-video-feed");
    
    if (uploadControls) uploadControls.style.display = "none";
    if (playerArea) playerArea.style.display = "block";
    if (feed) feed.src = `/play_video/${type}/${filename}`;
}

function stopVideo() {
    const uploadControls = document.getElementById("upload-controls");
    const playerArea = document.getElementById("video-player-area");
    const feed = document.getElementById("test-video-feed");
    
    if (feed) feed.src = "";
    if (playerArea) playerArea.style.display = "none";
    if (uploadControls) uploadControls.style.display = "block";
}

// Modal (Pop-Up Görsel) Ayarları
function openEvidenceModal(imgSrc, fileName) {
    const modal = document.getElementById("evidenceModal");
    const fullImg = document.getElementById("fullImg");
    const downloadBtn = document.getElementById("downloadBtn");
    
    if (modal && fullImg) {
        // pop-up modal flexbox olarak açılır ve ortalnır
        modal.style.display = "flex";
        modal.style.alignItems = "center";
        modal.style.justifyContent = "center";
        
        // arka plan kayması kiltlenir
        document.body.style.overflow = "hidden";
        
        fullImg.src = imgSrc;
        
        if (downloadBtn && fileName) {
            downloadBtn.href = imgSrc;
            downloadBtn.download = fileName;
        }
    }
}

function closeEvidenceModal() { 
    const modal = document.getElementById("evidenceModal");
    if (modal) {
        modal.style.display = "none"; 
        // arka plan kayma kilidi geri açılır
        document.body.style.overflow = "auto";
    }
}

window.onclick = function(event) {
    const modal = document.getElementById("evidenceModal");
    if (event.target === modal) {
        modal.style.display = "none";
        // arka plan kayma kilidi geri açılır
        document.body.style.overflow = "auto";
    }
}