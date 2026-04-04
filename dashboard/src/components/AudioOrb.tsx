import { useRef, useEffect, useCallback } from 'react'
import { useBrainStore } from '../stores/brainState'

export function AudioOrb() {
  const isAudioMode = useBrainStore((s) => s.isAudioMode)
  const audioState = useBrainStore((s) => s.audioState)

  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const recorder = new MediaRecorder(stream, { mimeType: 'audio/webm' })
      chunksRef.current = []

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data)
      }

      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' })
        stream.getTracks().forEach(t => t.stop())
        mediaRecorderRef.current = null
        useBrainStore.getState().submitAudio(blob)
      }

      mediaRecorderRef.current = recorder
      recorder.start()
      useBrainStore.getState().setAudioState('listening')
    } catch (err) {
      console.error('Mic access denied:', err)
      useBrainStore.getState().setAudioMode(false)
      useBrainStore.getState().setAudioState('idle')
    }
  }, [])

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop()
    }
  }, [])

  // Auto-start recording when audio mode is activated (e.g. from mic button)
  useEffect(() => {
    if (isAudioMode && !mediaRecorderRef.current) {
      startRecording()
    }
  }, [isAudioMode, startRecording])

  const handleClick = () => {
    if (isAudioMode) {
      stopRecording()
    }
  }

  if (!isAudioMode && audioState === 'idle') return null

  return (
    <div className="audio-orb-container" onClick={handleClick}>
      <div className={`audio-orb ${audioState}`}>
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
          <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/>
          <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
          <line x1="12" y1="19" x2="12" y2="22"/>
        </svg>
      </div>
      <div className="audio-orb-label">
        {audioState === 'listening' ? 'Listening...' : audioState === 'processing' ? 'Processing...' : ''}
      </div>
      <div className="waveform">
        {[1,2,3,4,5,6,7].map((n) => (
          <div key={n} className="waveform-bar" style={{ animation: `waveform${n} ${0.4 + n * 0.1}s ease-in-out infinite` }} />
        ))}
      </div>
    </div>
  )
}
