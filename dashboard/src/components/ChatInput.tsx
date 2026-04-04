import { useRef, useCallback } from 'react'
import { useBrainStore } from '../stores/brainState'
import { AudioOrb } from './AudioOrb'

const ACCEPT = 'image/*,video/*,application/pdf'

function addFilesToStore(files: FileList | File[]) {
  const store = useBrainStore.getState()
  Array.from(files).forEach((file) => {
    if (store.attachedFiles.some((f) => f.name === file.name && f.size === file.size)) return
    const url = URL.createObjectURL(file)
    store.addAttachedFile({ name: file.name, type: file.type, size: file.size, url })
  })
}

export { addFilesToStore }

export function ChatInput() {
  const text = useBrainStore((s) => s.chatInputText)
  const setChatInputText = useBrainStore((s) => s.setChatInputText)
  const isAudioMode = useBrainStore((s) => s.isAudioMode)
  const setAudioMode = useBrainStore((s) => s.setAudioMode)
  const setAudioState = useBrainStore((s) => s.setAudioState)
  const attachedFiles = useBrainStore((s) => s.attachedFiles)
  const removeAttachedFile = useBrainStore((s) => s.removeAttachedFile)
  const submitChat = useBrainStore((s) => s.submitChat)

  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleTextChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    setChatInputText(e.target.value)
  }, [setChatInputText])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    await submitChat()
  }

  const handleMicClick = () => {
    setAudioMode(true)
    setAudioState('listening')
  }

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) addFilesToStore(e.target.files)
    e.target.value = ''
  }

  return (
    <>
      <AudioOrb />
      {attachedFiles.length > 0 && (
        <div className="attached-files-bar">
          {attachedFiles.map((f) => (
            <div key={f.name} className="attached-chip">
              {f.type.startsWith('image/') ? (
                <img src={f.url} alt={f.name} className="attached-thumb" />
              ) : (
                <span className="attached-icon">
                  {f.type === 'application/pdf' ? (
                    <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Z"/><path d="M14 2v6h6"/></svg>
                  ) : (
                    <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2"><polygon points="23 7 16 12 23 17 23 7"/><rect x="1" y="5" width="15" height="14" rx="2"/></svg>
                  )}
                </span>
              )}
              <span className="attached-name">{f.name}</span>
              <button className="attached-remove" onClick={() => removeAttachedFile(f.name)}>&times;</button>
            </div>
          ))}
        </div>
      )}
      <div className={`input-bar${isAudioMode ? ' dimmed' : ''}`}>
        <form onSubmit={handleSubmit} className="input-container">
          <input
            className="input-field"
            type="text"
            value={text}
            onChange={handleTextChange}
            placeholder={isAudioMode ? 'Audio mode active...' : 'Ask anything...'}
            disabled={isAudioMode}
          />
          <input
            ref={fileInputRef}
            type="file"
            accept={ACCEPT}
            multiple
            style={{ display: 'none' }}
            onChange={handleFileSelect}
          />
          <button type="button" className="btn-memory" onClick={() => useBrainStore.getState().setMemoryPanelOpen(!useBrainStore.getState().memoryPanelOpen)}>
            <svg viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="10"/>
              <path d="M12 6v6l4 2"/>
            </svg>
          </button>
          <button type="button" className="btn-attach" onClick={() => fileInputRef.current?.click()}>
            <svg viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"/>
            </svg>
          </button>
          <button type="button" className="btn-mic" onClick={handleMicClick} disabled={isAudioMode}>
            <svg viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/>
              <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
              <line x1="12" y1="19" x2="12" y2="22"/>
            </svg>
          </button>
          <button type="submit" className="btn-send" disabled={isAudioMode || (!text.trim() && attachedFiles.length === 0)}>
            <svg viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="12" y1="19" x2="12" y2="5"/>
              <polyline points="5 12 12 5 19 12"/>
            </svg>
          </button>
        </form>
      </div>
    </>
  )
}
