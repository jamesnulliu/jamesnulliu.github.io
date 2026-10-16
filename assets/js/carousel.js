// Draggable carousel for the {{< carousel >}} shortcode.
(() => {
    const SWIPE_RATIO = 0.18;    // drag past this fraction of the width to change slide
    const FLICK_VELOCITY = 0.5;  // or flick faster than this (px/ms)
    const DRAG_THRESHOLD = 6;    // px moved before deciding drag vs. vertical scroll

    function init(root) {
        const track = root.querySelector(".carousel-track");
        const slides = Array.from(track.children);
        const dots = Array.from(root.querySelectorAll(".carousel-dot"));
        const count = slides.length;
        if (count < 2) return;

        const interval = parseInt(root.dataset.interval, 10) || 0;
        const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
        let index = 0;
        let timer = null;
        let drag = null;

        const render = (dx = 0) => {
            track.style.transform = `translate3d(calc(${-index * 100}% + ${dx}px), 0, 0)`;
        };

        const go = (i) => {
            index = (i + count) % count;
            render();
            dots.forEach((d, k) => d.setAttribute("aria-current", k === index ? "true" : "false"));
            // Lazy images clipped by overflow never load on their own; load neighbours ahead of time.
            for (const k of [index - 1, index, index + 1]) {
                const img = slides[(k + count) % count].querySelector("img");
                if (img) img.loading = "eager";
            }
        };

        const stop = () => {
            clearInterval(timer);
            timer = null;
        };
        const start = () => {
            stop();
            if (interval > 0 && !reduceMotion && !document.hidden) {
                timer = setInterval(() => go(index + 1), interval);
            }
        };
        const goAndRestart = (i) => {
            go(i);
            if (!root.matches(":hover, :focus-within")) start();
        };

        root.querySelector(".carousel-prev").addEventListener("click", () => goAndRestart(index - 1));
        root.querySelector(".carousel-next").addEventListener("click", () => goAndRestart(index + 1));
        dots.forEach((d, k) => d.addEventListener("click", () => goAndRestart(k)));
        root.addEventListener("keydown", (e) => {
            if (e.key === "ArrowLeft") goAndRestart(index - 1);
            else if (e.key === "ArrowRight") goAndRestart(index + 1);
        });

        root.addEventListener("mouseenter", stop);
        root.addEventListener("mouseleave", () => { if (!drag) start(); });
        root.addEventListener("focusin", stop);
        root.addEventListener("focusout", (e) => { if (!root.contains(e.relatedTarget)) start(); });
        document.addEventListener("visibilitychange", () => (document.hidden ? stop() : start()));

        track.addEventListener("pointerdown", (e) => {
            if (e.pointerType === "mouse" && e.button !== 0) return;
            drag = { id: e.pointerId, x0: e.clientX, y0: e.clientY, t0: e.timeStamp, dx: 0, active: false };
        });

        track.addEventListener("pointermove", (e) => {
            if (!drag || e.pointerId !== drag.id) return;
            const dx = e.clientX - drag.x0;
            const dy = e.clientY - drag.y0;
            if (!drag.active) {
                if (Math.abs(dx) < DRAG_THRESHOLD && Math.abs(dy) < DRAG_THRESHOLD) return;
                if (Math.abs(dy) > Math.abs(dx)) {
                    drag = null; // vertical gesture: let the page scroll
                    return;
                }
                drag.active = true;
                track.setPointerCapture(e.pointerId);
                root.classList.add("is-dragging");
                stop();
            }
            // Resist dragging past the first / last slide.
            const pastEdge = (index === 0 && dx > 0) || (index === count - 1 && dx < 0);
            drag.dx = pastEdge ? dx / 3 : dx;
            render(drag.dx);
        });

        const endDrag = (e) => {
            if (!drag || e.pointerId !== drag.id) return;
            const { active, dx, t0 } = drag;
            drag = null;
            if (!active) return;
            root.classList.remove("is-dragging");
            const velocity = dx / Math.max(1, e.timeStamp - t0);
            const far = Math.abs(dx) > root.clientWidth * SWIPE_RATIO;
            const flick = Math.abs(velocity) > FLICK_VELOCITY && Math.abs(dx) > 20;
            let next = index;
            if (far || flick) next = Math.min(count - 1, Math.max(0, index + (dx < 0 ? 1 : -1)));
            go(next);
            if (!root.matches(":hover")) start();
        };
        track.addEventListener("pointerup", endDrag);
        track.addEventListener("pointercancel", endDrag);

        go(0);
        start();
    }

    document.querySelectorAll(".carousel").forEach(init);
})();
