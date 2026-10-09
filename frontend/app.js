let currentFile = null;
let originalCorners = []; // In original image coords
let previewCorners = []; // In preview UI coords
let uiScale = 1.0; 
let baseScale = 1.0; // Scale from original to API preview
let imgNaturalSize = [0, 0];
let currentRotation = 0; // 0, 1, 2, 3 (quarter turns)
let activeHandle = null;

let straightenTimeout = null;
let straightenedResultBase64 = null;

document.addEventListener('DOMContentLoaded', () => {
    // Dropzone setup
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, e => { e.preventDefault(); e.stopPropagation(); });
    });
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'));
    });
    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'));
    });
    dropZone.addEventListener('drop', e => handleFiles(e.dataTransfer.files));
    fileInput.addEventListener('change', function() { handleFiles(this.files); });

    function handleFiles(files) {
        if (!files.length) return;
        currentFile = files[0];
        detectPage();
    }

    // Step Management
    function setStep(step) {
        document.querySelectorAll('.step-indicator .step').forEach(el => el.classList.remove('active'));
        document.querySelectorAll('.step-section').forEach(el => el.classList.remove('active'));
        document.getElementById(`ind-step${step}`).classList.add('active');
        if (step === 1) document.getElementById('step1-upload').classList.add('active');
        if (step === 2) document.getElementById('step2-straighten').classList.add('active');
        if (step === 3) document.getElementById('step3-enhance').classList.add('active');
        if (step === 4) document.getElementById('step4-geometry').classList.add('active');
        if (step === 5) document.getElementById('step5-dimensions').classList.add('active');
        if (step === 6) document.getElementById('step6-solve').classList.add('active');
    }

    function showLoader(text) {
        document.getElementById('global-loader-text').textContent = text;
        document.getElementById('global-loader').classList.remove('hidden');
    }
    function hideLoader() {
        document.getElementById('global-loader').classList.add('hidden');
    }

    // --- STEP 2: DETECT & STRAIGHTEN ---
    async function detectPage() {
        showLoader("Analyzing page...");
        const fd = new FormData();
        fd.append('file', currentFile);
        
        try {
            const res = await fetch('/api/detect-page', { method: 'POST', body: fd });
            const data = await res.json();
            if(!res.ok) throw new Error(data.error);

            baseScale = data.scale;
            imgNaturalSize = data.original_size;
            originalCorners = data.corners_orig;
            previewCorners = data.corners_preview; // Initial preview corners matching the API preview base64
            currentRotation = 0;

            const img = document.getElementById('preview-img');
            img.onload = () => {
                // Once image is rendered, calculate the UI scale relative to the API preview
                calcUiScale();
                drawPolygon();
                scheduleLivePreview();
                hideLoader();
                setStep(2);
            };
            img.src = data.preview;

            // Warnings
            const warnBox = document.getElementById('detection-warnings');
            if (data.warnings && data.warnings.length > 0) {
                warnBox.innerHTML = data.warnings.map(w => `<div class="warning-item">⚠️ ${w}</div>`).join('');
                warnBox.classList.remove('hidden');
            } else {
                warnBox.classList.add('hidden');
            }

        } catch (e) {
            hideLoader();
            alert(e.message);
        }
    }

    function calcUiScale() {
        const img = document.getElementById('preview-img');
        // img.naturalWidth is the size of the base64 preview returned by API
        // img.width is the rendered size in CSS
        if(img.naturalWidth > 0) {
            uiScale = img.width / img.naturalWidth;
        }
    }

    window.addEventListener('resize', () => {
        if(document.getElementById('step2-straighten').classList.contains('active')) {
            calcUiScale();
            drawPolygon();
        }
    });

    function getRenderedCorners() {
        // Returns corners in CSS pixels relative to the image
        return previewCorners.map(pt => [pt[0] * uiScale, pt[1] * uiScale]);
    }
    
    function updateOrigFromPreview() {
        // Convert previewCorners back to original image space
        originalCorners = previewCorners.map(pt => [pt[0] / baseScale, pt[1] / baseScale]);
    }

    function drawPolygon() {
        const img = document.getElementById('preview-img');
        const container = document.getElementById('canvas-container');
        
        // Offset for the image centering in the flex container
        const offsetX = (container.clientWidth - img.width) / 2;
        const offsetY = (container.clientHeight - img.height) / 2;

        const pts = getRenderedCorners();
        const polyStr = pts.map(p => `${p[0] + offsetX},${p[1] + offsetY}`).join(' ');
        document.getElementById('corner-polygon').setAttribute('points', polyStr);

        for (let i = 0; i < 4; i++) {
            const h = document.getElementById(`h-${['tl','tr','br','bl'][i]}`);
            h.setAttribute('cx', pts[i][0] + offsetX);
            h.setAttribute('cy', pts[i][1] + offsetY);
        }
    }

    // Interaction setup
    const svg = document.getElementById('corner-overlay');
    const handles = document.querySelectorAll('.handle');
    const loupe = document.getElementById('loupe');
    const img = document.getElementById('preview-img');

    function getMousePos(e) {
        const rect = svg.getBoundingClientRect();
        const clientX = e.touches ? e.touches[0].clientX : e.clientX;
        const clientY = e.touches ? e.touches[0].clientY : e.clientY;
        return {
            x: clientX - rect.left,
            y: clientY - rect.top
        };
    }

    function handleStart(e) {
        if(e.target.classList.contains('handle')) {
            e.preventDefault();
            activeHandle = e.target;
            loupe.classList.remove('hidden');
            loupe.style.backgroundImage = `url(${img.src})`;
            updateDrag(e);
        }
    }

    function handleMove(e) {
        if (!activeHandle) return;
        e.preventDefault();
        updateDrag(e);
    }

    function updateDrag(e) {
        const pos = getMousePos(e);
        const container = document.getElementById('canvas-container');
        const offsetX = (container.clientWidth - img.width) / 2;
        const offsetY = (container.clientHeight - img.height) / 2;

        // Clamp to image bounds
        let imgX = pos.x - offsetX;
        let imgY = pos.y - offsetY;
        imgX = Math.max(0, Math.min(imgX, img.width));
        imgY = Math.max(0, Math.min(imgY, img.height));

        const idx = parseInt(activeHandle.dataset.idx);
        
        // Update model
        previewCorners[idx] = [imgX / uiScale, imgY / uiScale];
        updateOrigFromPreview();
        drawPolygon();

        // Update loupe
        loupe.style.left = `${pos.x - 60}px`;
        loupe.style.top = `${pos.y - 140}px`; // offset above finger
        
        // Loupe background positioning
        // Need percentage based on image bounds
        const pctX = (imgX / img.width) * 100;
        const pctY = (imgY / img.height) * 100;
        loupe.style.backgroundPosition = `${pctX}% ${pctY}%`;
        loupe.style.backgroundSize = `${img.width * 2}px ${img.height * 2}px`; // 2x zoom

        scheduleLivePreview();
    }

    function handleEnd() {
        if(activeHandle) {
            activeHandle = null;
            loupe.classList.add('hidden');
        }
    }

    svg.addEventListener('mousedown', handleStart);
    svg.addEventListener('touchstart', handleStart, {passive: false});
    window.addEventListener('mousemove', handleMove);
    window.addEventListener('touchmove', handleMove, {passive: false});
    window.addEventListener('mouseup', handleEnd);
    window.addEventListener('touchend', handleEnd);

    // Sidebar Actions
    document.getElementById('btn-detect').addEventListener('click', detectPage);
    document.getElementById('btn-reset-corners').addEventListener('click', () => {
        // Set to full image
        const w = img.naturalWidth;
        const h = img.naturalHeight;
        const m = Math.max(w, h) * 0.05;
        previewCorners = [
            [m, m], [w-m, m], [w-m, h-m], [m, h-m]
        ];
        updateOrigFromPreview();
        drawPolygon();
        scheduleLivePreview();
    });

    document.getElementById('btn-rot-l').addEventListener('click', () => { currentRotation = (currentRotation + 3) % 4; scheduleLivePreview(); });
    document.getElementById('btn-rot-r').addEventListener('click', () => { currentRotation = (currentRotation + 1) % 4; scheduleLivePreview(); });
    document.getElementById('paper-select').addEventListener('change', scheduleLivePreview);

    function scheduleLivePreview() {
        clearTimeout(straightenTimeout);
        document.getElementById('live-preview-loader').classList.remove('hidden');
        straightenTimeout = setTimeout(fetchStraightenPreview, 400);
    }

    async function fetchStraightenPreview() {
        if(!currentFile) return;
        const fd = new FormData();
        // Downsample the image for live preview to be fast?
        // Actually, the API processes the full image. To be fast, we could send the preview base64 back,
        // but it's simpler to just call the API.
        fd.append('file', currentFile);
        fd.append('corners', JSON.stringify(originalCorners));
        fd.append('paper', document.getElementById('paper-select').value);
        fd.append('rotate', currentRotation);

        try {
            const res = await fetch('/api/straighten', { method: 'POST', body: fd });
            const data = await res.json();
            if(res.ok) {
                document.getElementById('live-preview-img').src = data.straightened;
                straightenedResultBase64 = data.straightened; // Cache for apply
            }
        } catch(e) {}
        finally {
            document.getElementById('live-preview-loader').classList.add('hidden');
        }
    }

    document.getElementById('btn-skip').addEventListener('click', () => {
        // Proceed to enhance with original image
        const reader = new FileReader();
        reader.onload = (e) => {
            straightenedResultBase64 = e.target.result;
            runEnhance();
        };
        reader.readAsDataURL(currentFile);
    });

    document.getElementById('btn-apply').addEventListener('click', async () => {
        if (!straightenedResultBase64) {
            showLoader("Straightening...");
            await fetchStraightenPreview();
            hideLoader();
        }
        if (straightenedResultBase64) {
            runEnhance();
        }
    });

    document.getElementById('link-back').addEventListener('click', (e) => {
        e.preventDefault();
        setStep(2);
    });

    // --- STEP 3: ENHANCE ---
    // Fetch directly from blob/base64?
    // /api/enhance expects a file upload. We must convert base64 back to file.
    function dataURLtoFile(dataurl, filename) {
        var arr = dataurl.split(','), mime = arr[0].match(/:(.*?);/)[1],
            bstr = atob(arr[1]), n = bstr.length, u8arr = new Uint8Array(n);
        while(n--){
            u8arr[n] = bstr.charCodeAt(n);
        }
        return new File([u8arr], filename, {type:mime});
    }

    async function runEnhance() {
        setStep(3);
        const statusArea = document.getElementById('status-area');
        const resultsSec = document.getElementById('results-section');
        const warnArea = document.getElementById('warnings-area');
        
        statusArea.classList.remove('hidden');
        resultsSec.classList.add('hidden');
        warnArea.classList.add('hidden');
        warnArea.innerHTML = '';

        const fd = new FormData();
        // straightenedResultBase64 contains the image
        const fileToEnhance = dataURLtoFile(straightenedResultBase64, "straightened.png");
        fd.append('file', fileToEnhance);
        
        try {
            const response = await fetch('/api/enhance', {
                method: 'POST',
                body: fd
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error);

            if (data.warnings && data.warnings.length > 0) {
                warnArea.classList.remove('hidden');
                data.warnings.forEach(w => {
                    const p = document.createElement('div');
                    p.className = 'warning-item';
                    p.textContent = `⚠️ ${w}`;
                    warnArea.appendChild(p);
                });
            }

            document.getElementById('img-original').src = data.original;
            document.getElementById('img-enhanced').src = data.enhanced;
            document.getElementById('img-binary').src = data.binary;
            
            document.getElementById('download-enhanced').href = data.enhanced;
            document.getElementById('download-binary').href = data.binary;

            document.getElementById('stats').textContent = `Processed ${data.image_size[0]}x${data.image_size[1]} in ${data.processing_time_s}s`;

            resultsSec.classList.remove('hidden');
        } catch (err) {
            alert(err.message);
            setStep(2);
        } finally {
            statusArea.classList.add('hidden');
        }
    }
    // --- STEP 4: GEOMETRY ---
    let geomData = null;
    let selectedSegmentId = null;

    document.getElementById('btn-continue-geometry').addEventListener('click', () => {
        runGeometry();
    });

    document.getElementById('link-back-enhance').addEventListener('click', (e) => {
        e.preventDefault();
        setStep(3);
    });

    async function runGeometry() {
        setStep(4);
        showLoader("Extracting geometry...");
        
        if(document.getElementById('step4-geometry')) {
            document.getElementById('step4-geometry').classList.add('active');
            document.getElementById('ind-step4').classList.add('active');
        }
        
        const binFile = dataURLtoFile(document.getElementById('img-binary').src, "binary.png");
        const enhFile = dataURLtoFile(document.getElementById('img-enhanced').src, "enhanced.png");
        
        const fd = new FormData();
        fd.append('binary', binFile);
        fd.append('enhanced', enhFile);
        
        try {
            const res = await fetch('/api/geometry', { method: 'POST', body: fd });
            geomData = await res.json();
            if(!res.ok) throw new Error(geomData.error);
            
            document.getElementById('geom-base-img').src = document.getElementById('img-enhanced').src;
            
            // Adjust SVG viewBox based on image size
            if(geomData.image) {
                document.getElementById('geom-svg').setAttribute("viewBox", `0 0 ${geomData.image.w} ${geomData.image.h}`);
            }
            
            renderGeometry();
        } catch(err) {
            alert(err.message);
            setStep(3);
        } finally {
            hideLoader();
        }
    }

    function renderGeometry() {
        if(!geomData) return;
        
        const layerSegments = document.getElementById('layer-segments');
        const layerVertices = document.getElementById('layer-vertices');
        const layerText = document.getElementById('layer-text');
        
        layerSegments.innerHTML = '';
        layerVertices.innerHTML = '';
        layerText.innerHTML = '';
        
        if (geomData.text_groups) {
            geomData.text_groups.forEach(tg => {
                const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
                rect.setAttribute("x", tg.bbox[0]);
                rect.setAttribute("y", tg.bbox[1]);
                rect.setAttribute("width", tg.bbox[2]);
                rect.setAttribute("height", tg.bbox[3]);
                rect.setAttribute("fill", "rgba(255, 255, 0, 0.2)");
                rect.setAttribute("stroke", "orange");
                layerText.appendChild(rect);
            });
        }

        const vMap = {};
        if (geomData.vertices) {
            geomData.vertices.forEach(v => {
                vMap[v.id] = v;
            });
        }

        if (geomData.segments) {
            geomData.segments.forEach(s => {
                const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
                const v1 = vMap[s.v1];
                const v2 = vMap[s.v2];
                if(!v1 || !v2) return;
                
                line.setAttribute("x1", v1.x);
                line.setAttribute("y1", v1.y);
                line.setAttribute("x2", v2.x);
                line.setAttribute("y2", v2.y);
                line.setAttribute("stroke-width", "3");
                
                let color = "grey";
                let dash = "";
                if (s.role === "outline") color = "blue";
                if (s.role === "feature") color = "green";
                if (s.role === "divider") color = "orange";
                if (s.role === "dimension" || s.role === "extension") {
                    color = "red";
                    dash = "4,4";
                }
                
                line.setAttribute("stroke", color);
                if(dash) line.setAttribute("stroke-dasharray", dash);
                
                line.style.cursor = "pointer";
                line.dataset.id = s.id;
                
                line.addEventListener('click', () => {
                    selectSegment(s.id);
                });
                
                layerSegments.appendChild(line);
            });
        }

        if (geomData.vertices) {
            geomData.vertices.forEach(v => {
                const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
                circle.setAttribute("cx", v.x);
                circle.setAttribute("cy", v.y);
                circle.setAttribute("r", "5");
                circle.setAttribute("fill", "white");
                circle.setAttribute("stroke", "blue");
                circle.setAttribute("stroke-width", "2");
                layerVertices.appendChild(circle);
            });
        }
        
        const nV = geomData.vertices ? geomData.vertices.length : 0;
        const nS = geomData.segments ? geomData.segments.length : 0;
        const nP = geomData.polygons ? geomData.polygons.length : 0;
        const nT = geomData.text_groups ? geomData.text_groups.length : 0;
        document.getElementById('summ-stats').textContent = `${nV} vertices, ${nS} segments, ${nP} polygons, ${nT} text blocks`;
    }

    function selectSegment(id) {
        selectedSegmentId = id;
        const s = geomData.segments.find(x => x.id === id);
        if(!s) return;
        
        document.getElementById('sel-len').textContent = `Len: ${Math.round(s.length_px)}px (${s.orientation})`;
        document.getElementById('sel-conf').textContent = `Conf: ${Math.round(s.confidence*100)}%`;
        document.getElementById('sel-role').value = s.role;
        document.getElementById('selection-panel').classList.remove('hidden');
    }

    document.getElementById('sel-role').addEventListener('change', async (e) => {
        if(!selectedSegmentId || !geomData) return;
        const s = geomData.segments.find(x => x.id === selectedSegmentId);
        if(s) {
            s.role = e.target.value;
            s.confidence = 1.0;
            // Send to recompute
            showLoader("Updating topology...");
            const fd = new FormData();
            fd.append('geometry_json', JSON.stringify(geomData));
            const res = await fetch('/api/geometry/overlay', { method: 'POST', body: fd });
            if(res.ok) {
                geomData = await res.json();
                renderGeometry();
            }
            hideLoader();
        }
    });

    document.getElementById('opacity-slider').addEventListener('input', (e) => {
        document.getElementById('geom-base-img').style.opacity = e.target.value / 100.0;
    });

    document.getElementById('btn-dl-json').addEventListener('click', () => {
        if(!geomData) return;
        const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(geomData, null, 2));
        const anchor = document.createElement('a');
        anchor.setAttribute("href", dataStr);
        anchor.setAttribute("download", "geometry.json");
        document.body.appendChild(anchor);
        anchor.click();
        anchor.remove();
    });
    
    // Toggles
    ['outline', 'feature', 'divider', 'dimension', 'text', 'vertices'].forEach(layer => {
        const t = document.getElementById(`toggle-${layer}`);
        if(t) {
            t.addEventListener('change', (e) => {
                // simple CSS toggle on the group could work, or full re-render
                renderGeometry(); // for simplicity, but ideally we'd toggle CSS classes
            });
        }
    });
    // --- STEP 5: DIMENSIONS ---
    let dimData = null;

    document.getElementById('btn-continue-dimensions').addEventListener('click', () => {
        runDimensions();
    });

    document.getElementById('link-back-geometry').addEventListener('click', (e) => {
        e.preventDefault();
        setStep(4);
    });

    document.getElementById('btn-rerun-ocr').addEventListener('click', () => {
        runDimensions();
    });

    async function runDimensions() {
        setStep(5);
        showLoader("Running OCR (Loading model may take ~10s on first run)...");
        
        if(!geomData) {
            hideLoader();
            return;
        }
        
        const enhFile = dataURLtoFile(document.getElementById('img-enhanced').src, "enhanced.png");
        const fd = new FormData();
        fd.append('geometry_json', JSON.stringify(geomData));
        fd.append('enhanced', enhFile);
        fd.append('engine', document.getElementById('sel-ocr-engine').value);
        fd.append('default_unit', document.getElementById('sel-default-unit').value);
        
        try {
            const res = await fetch('/api/dimensions', { method: 'POST', body: fd });
            dimData = await res.json();
            if(!res.ok) throw new Error(dimData.error);
            
            document.getElementById('dim-base-img').src = document.getElementById('img-enhanced').src;
            
            if(geomData.image) {
                document.getElementById('dim-svg').setAttribute("viewBox", `0 0 ${geomData.image.w} ${geomData.image.h}`);
            }
            
            renderDimensions();
            renderDimensionsTable();
        } catch(err) {
            alert(err.message);
            setStep(4);
        } finally {
            hideLoader();
        }
    }

    function renderDimensions() {
        if(!dimData || !geomData) return;
        
        const layerSegments = document.getElementById('dim-layer-segments');
        const layerText = document.getElementById('dim-layer-text');
        
        layerSegments.innerHTML = '';
        layerText.innerHTML = '';
        
        const vMap = {};
        geomData.vertices.forEach(v => vMap[v.id] = v);
        
        // Draw dimension lines
        geomData.segments.forEach(s => {
            if(s.role === "dimension" || s.role === "extension") {
                const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
                const v1 = vMap[s.v1];
                const v2 = vMap[s.v2];
                if(!v1 || !v2) return;
                
                line.setAttribute("x1", v1.x);
                line.setAttribute("y1", v1.y);
                line.setAttribute("x2", v2.x);
                line.setAttribute("y2", v2.y);
                line.setAttribute("stroke-width", "3");
                line.setAttribute("stroke", s.role === "dimension" ? "red" : "rgba(255,0,0,0.5)");
                
                // Check if this line is in dimData
                const isUsed = dimData.dimensions.some(d => d.dimension_line === s.id);
                if(isUsed) {
                    line.setAttribute("stroke-width", "5");
                    line.setAttribute("stroke", "blue");
                }
                
                layerSegments.appendChild(line);
            }
        });
        
        // Draw text boxes
        geomData.text_groups.forEach(tg => {
            const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
            rect.setAttribute("x", tg.bbox[0]);
            rect.setAttribute("y", tg.bbox[1]);
            rect.setAttribute("width", tg.bbox[2]);
            rect.setAttribute("height", tg.bbox[3]);
            
            const isUsed = dimData.dimensions.some(d => d.text_group === tg.id);
            const isUnpaired = dimData.unpaired_text_groups.includes(tg.id);
            
            if(isUsed) {
                rect.setAttribute("fill", "rgba(0, 0, 255, 0.2)");
                rect.setAttribute("stroke", "blue");
            } else if(isUnpaired) {
                rect.setAttribute("fill", "rgba(255, 165, 0, 0.2)");
                rect.setAttribute("stroke", "orange");
            } else {
                rect.setAttribute("fill", "none");
                rect.setAttribute("stroke", "gray");
            }
            layerText.appendChild(rect);
        });
    }

    function renderDimensionsTable() {
        if(!dimData) return;
        const tbody = document.getElementById('dim-tbody');
        tbody.innerHTML = '';
        
        let nRead = 0;
        let nCheck = 0;
        
        dimData.dimensions.forEach((d, idx) => {
            if(d.status !== "ignored") nRead++;
            if(d.status === "check" || d.status === "unreadable") nCheck++;
            
            const tr = document.createElement('tr');
            tr.dataset.idx = idx;
            
            const cropSrc = dimData.crops[d.text_group] || '';
            
            let altOptions = '<option value="">-</option>';
            if(d.alternatives) {
                d.alternatives.forEach(alt => {
                    altOptions += `<option value="${alt.text}">${alt.text}</option>`;
                });
            }
            
            tr.innerHTML = `
                <td><span class="status-badge ${d.status}">${d.status}</span></td>
                <td><img src="${cropSrc}" alt="crop" style="max-height:40px; border:1px solid #334155;"></td>
                <td><input type="text" value="${d.raw_text || ''}" class="dim-input-text"></td>
                <td class="parsed-val">
                    <div>${d.value_mm > 0 ? d.value_mm.toFixed(2) : '-'}</div>
                    <div style="font-size: 0.75rem; color: #64748b; margin-top: 4px;">
                        ${d.confidence > 0 ? (d.confidence * 100).toFixed(0) + '% | ' : ''}${d.variant ? d.variant + ' | ' : ''}${d.rotation !== undefined ? d.rotation + '°' : ''}<br>
                        ${d.process_time ? d.process_time.toFixed(1) + 's' : ''}
                    </div>
                </td>
                <td><select class="dim-alt-select">${altOptions}</select></td>
                <td>
                    <button class="btn-secondary dim-btn-ignore" style="padding: 4px; font-size: 0.8rem;">Ignore</button>
                </td>
            `;
            tbody.appendChild(tr);
            
            // Event listeners
            tr.querySelector('.dim-input-text').addEventListener('change', async (e) => {
                const val = e.target.value;
                const fd = new FormData();
                fd.append('text', val);
                fd.append('default_unit', document.getElementById('sel-default-unit').value);
                const res = await fetch('/api/dimensions/parse', { method: 'POST', body: fd });
                if(res.ok) {
                    const data = await res.json();
                    if(data.status === "ok") {
                        d.raw_text = val;
                        d.value_mm = data.parsed.value_mm;
                        d.unit_used = data.parsed.unit_used;
                        d.kind = data.parsed.kind;
                        d.status = "ok";
                        d.user_edited = true;
                        recomputeSheet();
                    } else {
                        d.status = "check";
                        recomputeSheet();
                    }
                }
            });
            
            tr.querySelector('.dim-alt-select').addEventListener('change', (e) => {
                if(e.target.value) {
                    tr.querySelector('.dim-input-text').value = e.target.value;
                    tr.querySelector('.dim-input-text').dispatchEvent(new Event('change'));
                }
            });
            
            tr.querySelector('.dim-btn-ignore').addEventListener('click', () => {
                if(d.status === "ignored") {
                    d.status = "check";
                } else {
                    d.status = "ignored";
                }
                recomputeSheet();
            });
        });
        
        // Update summary
        document.getElementById('dim-summary-banner').textContent = `${nRead} dimensions read, ${nCheck} need checking.`;
        
        // Update warnings
        const warnBox = document.getElementById('dim-warnings');
        if (dimData.warnings && dimData.warnings.length > 0) {
            warnBox.innerHTML = dimData.warnings.map(w => `<div class="warning-item">⚠️ ${w}</div>`).join('');
            warnBox.classList.remove('hidden');
        } else {
            warnBox.classList.add('hidden');
        }
    }
    
    async function recomputeSheet() {
        const fd = new FormData();
        fd.append('dimensions_json', JSON.stringify(dimData));
        const res = await fetch('/api/dimensions/recompute', { method: 'POST', body: fd });
        if(res.ok) {
            dimData = await res.json();
            renderDimensionsTable();
        }
    }

    document.getElementById('btn-dl-dim-json').addEventListener('click', () => {
        if(!dimData) return;
        const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(dimData, null, 2));
        const anchor = document.createElement('a');
        anchor.setAttribute("href", dataStr);
        anchor.setAttribute("download", "dimensions.json");
        document.body.appendChild(anchor);
        anchor.click();
        anchor.remove();
    });

    // --- STEP 6: SOLVE ---
    let modelData = null;
    let solveOptions = {
        origin: "bottom_left",
        mirror: { horizontal: false, vertical: false },
        user_constraints: [],
        accept_inferred: [],
        user_intents: {}
    };

    document.getElementById('btn-continue-solve').addEventListener('click', () => {
        runSolve();
    });

    document.getElementById('link-back-dimensions').addEventListener('click', (e) => {
        e.preventDefault();
        setStep(5);
    });
    
    document.getElementById('sel-origin').addEventListener('change', (e) => {
        solveOptions.origin = e.target.value;
        reRunSolve();
    });
    
    document.getElementById('chk-mirror-h').addEventListener('change', (e) => {
        solveOptions.mirror.horizontal = e.target.checked;
        reRunSolve();
    });
    
    document.getElementById('chk-mirror-v').addEventListener('change', (e) => {
        solveOptions.mirror.vertical = e.target.checked;
        reRunSolve();
    });
    
    document.getElementById('btn-reset-solve').addEventListener('click', () => {
        solveOptions.user_constraints = [];
        solveOptions.accept_inferred = [];
        solveOptions.user_intents = {};
        reRunSolve();
    });

    async function runSolve() {
        setStep(6);
        showLoader("Solving metric constraints...");
        
        if(!geomData || !dimData) {
            hideLoader();
            return;
        }
        
        try {
            const res = await fetch('/api/solve', { 
                method: 'POST', 
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    geometry: geomData,
                    dimensions: dimData,
                    options: solveOptions
                })
            });
            modelData = await res.json();
            if(!res.ok) throw new Error(modelData.error || "Solve failed");
            
            renderSolve();
        } catch(err) {
            alert(err.message);
            setStep(5);
        } finally {
            hideLoader();
        }
    }
    
    async function reRunSolve() {
        showLoader("Updating solution...");
        try {
            const res = await fetch('/api/solve/edit', { 
                method: 'POST', 
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    geometry: geomData,
                    dimensions: dimData,
                    options: solveOptions
                })
            });
            modelData = await res.json();
            if(!res.ok) throw new Error(modelData.error || "Solve edit failed");
            
            renderSolve();
        } catch(err) {
            alert(err.message);
        } finally {
            hideLoader();
        }
    }

    function renderSolve() {
        if(!modelData) return;
        
        // Status Banner
        const banner = document.getElementById('solve-status-banner');
        if (modelData.dof.status === "fully_constrained") {
            banner.textContent = "Fully Constrained Model";
            banner.style.background = "#d1fae5";
            banner.style.color = "#065f46";
        } else if (modelData.dof.status === "conflicting") {
            banner.textContent = "Conflicts Detected";
            banner.style.background = "#fee2e2";
            banner.style.color = "#991b1b";
        } else {
            banner.textContent = `Under-constrained (${modelData.estimates.length} estimated values)`;
            banner.style.background = "#fef3c7";
            banner.style.color = "#92400e";
        }
        
        // Conflicts
        const conflictsBox = document.getElementById('solve-conflicts');
        if (modelData.conflicts && modelData.conflicts.length > 0) {
            let html = "<strong>Inconsistent Loop:</strong><br>";
            modelData.conflicts.forEach(c => {
                html += `Dimensions: ${c.dimensions.join(", ")} (Residuals: ${c.residuals_mm.map(r=>r.toFixed(2)).join(", ")} mm)<br>`;
            });
            conflictsBox.innerHTML = html;
            conflictsBox.classList.remove('hidden');
        } else {
            conflictsBox.classList.add('hidden');
        }
        
        // Estimates
        const estBox = document.getElementById('solve-estimates');
        if (modelData.estimates && modelData.estimates.length > 0) {
            let html = "<strong>Unconstrained Features:</strong><ul style='margin: 5px 0; padding-left: 20px;'>";
            modelData.estimates.forEach(est => {
                html += `<li>Edge ${est.target}: ~${est.value_mm.toFixed(1)} mm 
                <button class="btn-confirm-est" data-id="${est.id}" data-val="${est.value_mm}" data-target="${est.target}" data-axis="${est.axis}">Confirm</button></li>`;
            });
            html += "</ul>";
            estBox.innerHTML = html;
            estBox.classList.remove('hidden');
            
            document.querySelectorAll('.btn-confirm-est').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    const val = parseFloat(e.target.dataset.val);
                    const axis = e.target.dataset.axis;
                    // Ask user to confirm value
                    const userValStr = prompt(`Confirm size (mm):`, val);
                    if (userValStr !== null) {
                        const userVal = parseFloat(userValStr);
                        if(!isNaN(userVal) && userVal > 0) {
                            // Find the edge points in geometry to make a constraint
                            const targetId = e.target.dataset.target;
                            const s = geomData.segments.find(sg => sg.id === targetId);
                            if (s) {
                                // Add user constraint
                                // Simplified approach: just require the solve to add a user constraint on the same classes
                                const c1 = geomData.vertices.find(v => v.id === s.v1);
                                const c2 = geomData.vertices.find(v => v.id === s.v2);
                                solveOptions.user_constraints.push({
                                    id: `uc_${Date.now()}`,
                                    type: "distance",
                                    axis: axis,
                                    a: axis === "x" ? c1.id : c1.id, // backend constraints.py needs to handle this properly, but simplified here
                                    b: axis === "x" ? c2.id : c2.id,
                                    value_mm: userVal,
                                    source: "user",
                                    weight: 10.0
                                });
                                reRunSolve();
                            }
                        }
                    }
                });
            });
        } else {
            estBox.classList.add('hidden');
        }
        
        // Populate table
        const tbody = document.getElementById('solve-tbody');
        tbody.innerHTML = '';
        modelData.constraints.forEach(c => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${c.text || c.dimension || c.id}</td>
                <td>${c.value_mm.toFixed(2)}</td>
                <td>${c.residual_mm !== undefined ? c.residual_mm.toFixed(3) : '-'}</td>
                <td>${c.source}</td>
                <td></td>
            `;
            tbody.appendChild(tr);
        });
        
        // Draw Metric SVG
        const g = document.getElementById('solve-zoom-g');
        g.innerHTML = '';
        
        let minX = 0, minY = 0, maxX = 10, maxY = 10;
        if (modelData.vertices.length > 0) {
            minX = Math.min(...modelData.vertices.map(v => v.x));
            minY = Math.min(...modelData.vertices.map(v => v.y));
            maxX = Math.max(...modelData.vertices.map(v => v.x));
            maxY = Math.max(...modelData.vertices.map(v => v.y));
        }
        
        const w = Math.max(maxX - minX, 10);
        const h = Math.max(maxY - minY, 10);
        const padding = Math.max(w, h) * 0.1;
        
        // SVG viewBox should account for Y being up now
        document.getElementById('solve-svg').setAttribute("viewBox", `${minX - padding} ${-maxY - padding} ${w + 2*padding} ${h + 2*padding}`);
        
        const vMap = {};
        modelData.vertices.forEach(v => vMap[v.id] = v);
        
        modelData.segments.forEach(s => {
            const v1 = vMap[s.v1];
            const v2 = vMap[s.v2];
            if(v1 && v2) {
                const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
                line.setAttribute("x1", v1.x);
                line.setAttribute("y1", -v1.y); // SVG y is down, model y is up
                line.setAttribute("x2", v2.x);
                line.setAttribute("y2", -v2.y);
                
                let stroke = "black";
                let strokeDasharray = "none";
                if (v1.estimated || v2.estimated) {
                    stroke = "orange";
                    strokeDasharray = "4,2";
                }
                
                line.setAttribute("stroke", stroke);
                line.setAttribute("stroke-width", Math.max(w,h) * 0.005);
                line.setAttribute("stroke-dasharray", strokeDasharray);
                
                if (s.role === "divider") {
                    line.setAttribute("stroke", "blue");
                    line.setAttribute("stroke-dasharray", "2,2");
                }
                
                g.appendChild(line);
            }
        });
        
        if (modelData.dof.status !== "conflicting") {
            document.getElementById('btn-export-dxf').disabled = false;
        } else {
            document.getElementById('btn-export-dxf').disabled = true;
        }
    }

    document.getElementById('btn-dl-model').addEventListener('click', () => {
        if(!modelData) return;
        const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(modelData, null, 2));
        const anchor = document.createElement('a');
        anchor.setAttribute("href", dataStr);
        anchor.setAttribute("download", "model.json");
        document.body.appendChild(anchor);
        anchor.click();
        anchor.remove();
    });
});
