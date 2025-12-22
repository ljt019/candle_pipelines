use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

/// A handle to a registered XML tag
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Tag {
    name: String,
    id: usize,
}

impl Tag {
    /// Get the name of this tag
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the unique ID of this tag
    pub fn id(&self) -> usize {
        self.id
    }
}

impl PartialEq<str> for Tag {
    fn eq(&self, other: &str) -> bool {
        self.name == other
    }
}

impl PartialEq<&str> for Tag {
    fn eq(&self, other: &&str) -> bool {
        self.name == *other
    }
}

impl PartialEq<String> for Tag {
    fn eq(&self, other: &String) -> bool {
        self.name == *other
    }
}

impl PartialEq<&Tag> for Tag {
    fn eq(&self, other: &&Tag) -> bool {
        self == *other
    }
}

/// Parts of a tag emitted as events during parsing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TagParts {
    /// Opening of a tag
    Start,
    /// Content inside a tag or outside any tag
    Content,
    /// Closing of a tag
    End,
}

/// An event emitted by the XML parser
#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    /// Event originating from a registered tag
    Tagged {
        tag: Tag,
        part: TagParts,
        content: String,
    },
    /// Plain text outside any registered tag
    Output { part: TagParts, content: String },
}

impl Event {
    fn tagged(tag: Tag, part: TagParts, content: impl Into<String>) -> Self {
        Self::Tagged {
            tag,
            part,
            content: content.into(),
        }
    }

    fn plain(part: TagParts, content: impl Into<String>) -> Self {
        Self::Output {
            part,
            content: content.into(),
        }
    }

    /// Create a new content event outside any tag
    pub fn content(content: impl Into<String>) -> Self {
        Self::plain(TagParts::Content, content)
    }

    /// Create a new start event for top-level content
    pub fn plain_start() -> Self {
        Self::plain(TagParts::Start, "")
    }

    /// Create a new end event for top-level content
    pub fn plain_end() -> Self {
        Self::plain(TagParts::End, "")
    }

    /// Create a new start event for a tag
    pub fn start(tag: Tag) -> Self {
        Self::tagged(tag, TagParts::Start, "")
    }

    /// Create a new end event for a tag
    pub fn end(tag: Tag) -> Self {
        Self::tagged(tag, TagParts::End, "")
    }

    /// Create a new content event inside a tag
    fn tagged_internal(tag: Tag, content: impl Into<String>) -> Self {
        Self::tagged(tag, TagParts::Content, content)
    }

    /// Get the content string
    pub fn get_content(&self) -> &str {
        match self {
            Self::Tagged { content, .. } | Self::Output { content, .. } => content,
        }
    }

    /// Get the tag name if present
    pub fn tag(&self) -> Option<&str> {
        match self {
            Self::Tagged { tag, .. } => Some(tag.name()),
            Self::Output { .. } => None,
        }
    }

    /// Get the internal tag handle
    pub fn tag_handle(&self) -> Option<&Tag> {
        match self {
            Self::Tagged { tag, .. } => Some(tag),
            Self::Output { .. } => None,
        }
    }

    /// Get the part of the tag this event corresponds to
    pub fn part(&self) -> TagParts {
        match self {
            Self::Tagged { part, .. } | Self::Output { part, .. } => *part,
        }
    }
}

/// Builder for creating an XmlParser with specific tags to watch for
#[derive(Debug, Default)]
pub struct XmlParserBuilder {
    tags: Vec<String>,
    next_id: usize,
}

impl XmlParserBuilder {
    /// Create a new XmlParserBuilder
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a tag to watch for during parsing and return a handle to it
    pub fn register_tag(&mut self, tag: impl Into<String>) -> Tag {
        let name = tag.into();
        let tag_handle = Tag {
            name: name.clone(),
            id: self.next_id,
        };
        self.next_id += 1;
        self.tags.push(name);
        tag_handle
    }

    /// Build the XmlParser
    pub fn build(self) -> XmlParser {
        let mut tag_map = HashMap::new();
        let mut tags_set = HashSet::new();

        for (id, name) in self.tags.into_iter().enumerate() {
            tags_set.insert(name.clone());
            tag_map.insert(name.clone(), Tag { name, id });
        }

        XmlParser::new(tags_set, tag_map)
    }
}

/// Parser state for tracking XML tag parsing
#[derive(Debug, Clone, Default)]
struct ParserState {
    /// Stack of currently open tags (tag name and accumulated content)
    open_tags: Vec<(String, String)>,
    /// Buffer for content outside any registered tags
    content_buffer: String,
    /// Buffer for partial tag parsing
    tag_buffer: String,
    /// Whether we're currently inside a tag
    in_tag: bool,
    /// Length of top-level content that has already been emitted downstream. This lets us
    /// stream only the newly arrived slice on every call to `parse_token`.
    emitted_top_len: usize,
    /// For each open tag name we keep the number of characters that have already been
    /// emitted so we can stream incremental updates without duplication.
    emitted_tag_lens: std::collections::HashMap<String, usize>,
    /// Whether we have started emitting top-level content
    top_level_open: bool,
    /// Whether the last emitted top-level content ended with a newline
    last_content_had_newline: bool,
}

/// XML parser that can process streaming or complete text and emit events
#[derive(Debug, Clone)]
pub struct XmlParser {
    /// Set of tags this parser is configured to watch for
    registered_tags: HashSet<String>,
    /// Map from tag names to Tag handles
    tag_map: HashMap<String, Tag>,
    /// Current parser state (wrapped for interior mutability)
    state: Arc<Mutex<ParserState>>,
}

impl XmlParser {
    /// Create a new XmlParser with the specified tags to watch for
    pub fn new(tags: HashSet<String>, tag_map: HashMap<String, Tag>) -> Self {
        Self {
            registered_tags: tags,
            tag_map,
            state: Arc::new(Mutex::new(ParserState::default())),
        }
    }

    /// Reset the parser state (useful for new generations)
    pub fn reset(&self) {
        *self.state.lock().expect("parser lock poisoned") = ParserState::default();
    }

    /// Parse a complete text and return all events
    pub fn parse_complete(&self, text: &str) -> Vec<Event> {
        self.reset();
        let mut events = Vec::new();

        for char in text.chars() {
            let mut evs = self.process_char(char);
            events.append(&mut evs);
        }

        events.extend(self.flush());
        events
    }

    /// Parse a streaming token and return any events that are ready
    pub fn parse_token(&self, token: &str) -> Vec<Event> {
        let mut events = Vec::new();

        for char in token.chars() {
            let mut evs = self.process_char(char);
            events.append(&mut evs);
        }

        // Emit the newly appended slice either for top-level content or for the currently
        // innermost open tag (if any). This enables true streaming behaviour: callers get
        // incremental updates instead of the whole block at close time.
        {
            let mut state = self.state.lock().expect("parser lock poisoned");

            if state.open_tags.is_empty() {
                // Outside of any registered tag.
                let current_len = state.content_buffer.len();
                if current_len > state.emitted_top_len {
                    let mut new_slice = &state.content_buffer[state.emitted_top_len..];

                    // If this is the very first top-level content emission, strip leading newlines.
                    if state.emitted_top_len == 0 {
                        new_slice = new_slice.trim_start_matches('\n');
                    }

                    // Skip emitting if it is now empty or just whitespace/newlines.
                    let content_to_emit = if new_slice.trim().is_empty() {
                        "".to_string()
                    } else {
                        new_slice.to_string()
                    };

                    if !content_to_emit.is_empty() {
                        if !state.top_level_open {
                            events.push(Event::plain_start());
                            state.top_level_open = true;
                        }
                        events.push(Event::content(content_to_emit.clone()));
                        state.last_content_had_newline = content_to_emit.ends_with('\n');
                    }
                    state.emitted_top_len = current_len;
                }
            } else {
                // Inside the innermost registered tag → stream its delta.
                if let Some((tag_name_ref, content_ref)) = state.open_tags.last() {
                    let tag_name = tag_name_ref.clone();
                    let content = content_ref.clone();
                    let total_len = content.len();

                    let already_emitted = *state.emitted_tag_lens.get(&tag_name).unwrap_or(&0);

                    if total_len > already_emitted {
                        let new_slice = &content[already_emitted..];

                        // Strip leading newlines from the first emission of tag content
                        if already_emitted == 0 {
                            let trimmed = new_slice.trim_start_matches('\n');
                            if !trimmed.is_empty() {
                                if let Some(tag_handle) = self.tag_map.get(&tag_name) {
                                    events
                                        .push(Event::tagged_internal(tag_handle.clone(), trimmed));
                                }
                            }
                            // Update emitted length to account for any trimmed newlines
                            state.emitted_tag_lens.insert(tag_name.clone(), total_len);
                        } else {
                            // Not the first emission, emit as-is
                            if let Some(tag_handle) = self.tag_map.get(&tag_name) {
                                events.push(Event::tagged_internal(tag_handle.clone(), new_slice));
                            }
                            state.emitted_tag_lens.insert(tag_name.clone(), total_len);
                        }
                    }
                }
            }
        }

        events
    }

    /// Process a single character and return an event if one is ready
    fn process_char(&self, c: char) -> Vec<Event> {
        let mut events = Vec::new();
        let mut state = self.state.lock().expect("parser lock poisoned");

        match c {
            '<' => {
                state.in_tag = true;
                state.tag_buffer.clear();
                state.tag_buffer.push(c);
            }
            '>' if state.in_tag => {
                state.tag_buffer.push(c);
                state.in_tag = false;

                let tag_content = state.tag_buffer.clone();
                state.tag_buffer.clear();

                events.extend(self.handle_tag(&mut state, &tag_content));
            }
            _ if state.in_tag => {
                state.tag_buffer.push(c);
            }
            _ => {
                if let Some((_, ref mut content)) = state.open_tags.last_mut() {
                    content.push(c);
                } else {
                    state.content_buffer.push(c);
                }
            }
        }

        events
    }

    /// Handle a complete tag and return an event if one is ready
    fn handle_tag(&self, state: &mut ParserState, tag_content: &str) -> Vec<Event> {
        let mut events = Vec::new();

        if let Some(tag_name) = self.parse_tag_name(tag_content) {
            if self.registered_tags.contains(&tag_name) {
                if tag_content.starts_with("</") {
                    if let Some(pos) = state
                        .open_tags
                        .iter()
                        .rposition(|(name, _)| name == &tag_name)
                    {
                        let (_, content) = state.open_tags.remove(pos);

                        let already_emitted = state.emitted_tag_lens.remove(&tag_name).unwrap_or(0);

                        if let Some(tag_handle) = self.tag_map.get(&tag_name) {
                            if content.len() > already_emitted {
                                let remaining_content = &content[already_emitted..];
                                // Strip leading newlines if this is the first content emission
                                let content_to_emit = if already_emitted == 0 {
                                    remaining_content.trim_start_matches('\n')
                                } else {
                                    remaining_content
                                };

                                // Also strip trailing newlines from the final emission
                                let trimmed = content_to_emit.trim_end_matches('\n');
                                if !trimmed.is_empty() {
                                    let mut final_str = trimmed.to_string();
                                    final_str.push('\n'); // ensure exactly one trailing newline
                                    events.push(Event::tagged_internal(
                                        tag_handle.clone(),
                                        final_str,
                                    ));
                                }
                            }
                            events.push(Event::end(tag_handle.clone()));
                        }
                    }
                } else if !tag_content.ends_with("/>") {
                    if state.open_tags.is_empty() && !state.content_buffer.is_empty() {
                        let content = &state.content_buffer[state.emitted_top_len..];

                        // Normalize whitespace before tag start
                        let mut slice = content;
                        if state.emitted_top_len == 0 {
                            slice = slice.trim_start_matches('\n');
                        }
                        let content_to_emit = if slice.trim().is_empty() {
                            String::new()
                        } else {
                            // Ensure top-level content ends with exactly one newline
                            let mut content_str = slice.to_string();
                            if !content_str.ends_with('\n') {
                                content_str.push('\n');
                            }
                            content_str
                        };

                        state.emitted_top_len = state.content_buffer.len();
                        if !content_to_emit.is_empty() {
                            if !state.top_level_open {
                                events.push(Event::plain_start());
                                state.top_level_open = true;
                            }
                            events.push(Event::content(content_to_emit.clone()));
                            state.last_content_had_newline = content_to_emit.ends_with('\n');
                        }
                    }

                    state.open_tags.push((tag_name.clone(), String::new()));

                    if let Some(tag_handle) = self.tag_map.get(&tag_name) {
                        events.push(Event::start(tag_handle.clone()));
                    }
                }
            } else if state.open_tags.is_empty() {
                state.content_buffer.push_str(tag_content);
            } else if let Some((_, ref mut content)) = state.open_tags.last_mut() {
                content.push_str(tag_content);
            }
        } else if state.open_tags.is_empty() {
            state.content_buffer.push_str(tag_content);
        } else if let Some((_, ref mut content)) = state.open_tags.last_mut() {
            content.push_str(tag_content);
        }

        events
    }

    /// Extract tag name from tag content (e.g., "<think>" -> "think", "</think>" -> "think")
    fn parse_tag_name(&self, tag_content: &str) -> Option<String> {
        if tag_content.len() < 3 || !tag_content.starts_with('<') || !tag_content.ends_with('>') {
            return None;
        }

        let inner = &tag_content[1..tag_content.len() - 1];

        if let Some(name) = inner.strip_prefix('/') {
            // Closing tag
            Some(name.split_whitespace().next()?.to_string())
        } else {
            // Opening tag or self-closing tag
            let name = inner.split_whitespace().next()?;
            if let Some(stripped) = name.strip_suffix('/') {
                Some(stripped.to_string())
            } else {
                Some(name.to_string())
            }
        }
    }

    /// Flush any remaining content and return events
    pub fn flush(&self) -> Vec<Event> {
        let mut state = self.state.lock().expect("parser lock poisoned");
        let mut events = Vec::new();

        // Emit any remaining content
        if state.content_buffer.len() > state.emitted_top_len {
            let remaining = &state.content_buffer[state.emitted_top_len..];

            // Always trim leading and trailing newlines from final content
            let slice = remaining.trim_start_matches('\n').trim_end_matches('\n');

            let content_to_emit = if slice.trim().is_empty() {
                String::new()
            } else {
                // Ensure top-level content ends with exactly one newline
                let mut content_str = slice.to_string();
                if !content_str.ends_with('\n') {
                    content_str.push('\n');
                }
                content_str
            };

            state.emitted_top_len = state.content_buffer.len();
            if !content_to_emit.is_empty() {
                if !state.top_level_open {
                    events.push(Event::plain_start());
                    state.top_level_open = true;
                }
                events.push(Event::content(content_to_emit.clone()));
                state.last_content_had_newline = content_to_emit.ends_with('\n');
            }
        }
        if state.top_level_open {
            // For streaming case: add a trailing newline if the last content didn't have one
            if !state.last_content_had_newline {
                events.push(Event::content("\n"));
            }
            events.push(Event::plain_end());
        }
        state.top_level_open = false;
        state.content_buffer.clear();
        state.emitted_top_len = 0;

        // Emit any unclosed tags as content (malformed XML). We must take ownership of the
        // drained tags first to avoid double-borrowing `state`.
        let drained: Vec<_> = state.open_tags.drain(..).collect();
        for (tag_name, content) in drained {
            let already_emitted = state.emitted_tag_lens.remove(&tag_name).unwrap_or(0);

            if let Some(tag_handle) = self.tag_map.get(&tag_name) {
                if content.len() > already_emitted {
                    let remaining_content = &content[already_emitted..];
                    // Strip leading newlines if this is the first content emission
                    let content_to_emit = if already_emitted == 0 {
                        remaining_content.trim_start_matches('\n')
                    } else {
                        remaining_content
                    };

                    // Also strip trailing newlines from the final emission
                    let trimmed = content_to_emit.trim_end_matches('\n');
                    if !trimmed.is_empty() {
                        let mut final_str = trimmed.to_string();
                        final_str.push('\n'); // ensure exactly one trailing newline
                        events.push(Event::tagged_internal(tag_handle.clone(), final_str));
                    }
                }
                events.push(Event::end(tag_handle.clone()));
            }
        }

        events
    }

    /// Get the currently registered tags
    pub fn registered_tags(&self) -> &HashSet<String> {
        &self.registered_tags
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_parsing() {
        let mut builder = XmlParserBuilder::new();
        let think_tag = builder.register_tag("think");
        let parser = builder.build();

        let text = "<think>Hello world</think>Regular content";
        let events = parser.parse_complete(text);

        assert_eq!(events.len(), 6);
        assert_eq!(events[0].part(), TagParts::Start);
        assert_eq!(events[0].tag_handle(), Some(&think_tag));
        assert_eq!(events[1].part(), TagParts::Content);
        assert_eq!(events[1].get_content(), "Hello world\n");
        assert_eq!(events[2].part(), TagParts::End);
        assert_eq!(events[2].tag_handle(), Some(&think_tag));
        assert_eq!(events[3].part(), TagParts::Start);
        assert_eq!(events[3].tag_handle(), None);
        assert_eq!(events[4].part(), TagParts::Content);
        assert_eq!(events[4].get_content(), "Regular content\n");
        assert_eq!(events[5].part(), TagParts::End);
        assert_eq!(events[5].tag_handle(), None);
    }

    #[test]
    fn test_unregistered_tags() {
        let mut builder = XmlParserBuilder::new();
        let think_tag = builder.register_tag("think");
        let parser = builder.build();

        let text = "<think>Registered</think><other>Not registered</other>";
        let events = parser.parse_complete(text);

        assert_eq!(events.len(), 6);
        assert_eq!(events[0].part(), TagParts::Start);
        assert_eq!(events[0].tag_handle(), Some(&think_tag));
        assert_eq!(events[1].part(), TagParts::Content);
        assert_eq!(events[1].get_content(), "Registered\n");
        assert_eq!(events[2].part(), TagParts::End);
        assert_eq!(events[2].tag_handle(), Some(&think_tag));
        assert_eq!(events[3].part(), TagParts::Start);
        assert_eq!(events[3].tag_handle(), None);
        assert_eq!(events[4].part(), TagParts::Content);
        assert_eq!(events[4].get_content(), "<other>Not registered</other>\n");
        assert_eq!(events[5].part(), TagParts::End);
        assert_eq!(events[5].tag_handle(), None);
    }

    #[test]
    fn test_malformed_xml() {
        let mut builder = XmlParserBuilder::new();
        let think_tag = builder.register_tag("think");
        let parser = builder.build();

        let text = "<think>Unclosed tag content";
        let events = parser.parse_complete(text);

        assert_eq!(events.len(), 3);
        assert_eq!(events[0].part(), TagParts::Start);
        assert_eq!(events[0].tag_handle(), Some(&think_tag));
        assert_eq!(events[1].part(), TagParts::Content);
        assert_eq!(events[1].get_content(), "Unclosed tag content\n");
        assert_eq!(events[2].part(), TagParts::End);
        assert_eq!(events[2].tag_handle(), Some(&think_tag));
    }

    #[test]
    fn test_self_closing_tag_ignored() {
        let mut builder = XmlParserBuilder::new();
        builder.register_tag("think");
        let parser = builder.build();

        let text = "<think/>Hello";
        let events = parser.parse_complete(text);

        assert_eq!(events.len(), 3);
        assert_eq!(events[0].part(), TagParts::Start);
        assert_eq!(events[0].tag_handle(), None);
        assert_eq!(events[1].part(), TagParts::Content);
        assert_eq!(events[1].get_content(), "Hello\n");
        assert_eq!(events[2].part(), TagParts::End);
        assert_eq!(events[2].tag_handle(), None);
    }

    #[test]
    fn test_tag_with_attributes() {
        let mut builder = XmlParserBuilder::new();
        let think_tag = builder.register_tag("think");
        let parser = builder.build();

        let text = "<think answer=\"yes\">Content</think>";
        let events = parser.parse_complete(text);

        assert_eq!(events.len(), 3);
        assert_eq!(events[0].part(), TagParts::Start);
        assert_eq!(events[0].tag_handle(), Some(&think_tag));
        assert_eq!(events[1].part(), TagParts::Content);
        assert_eq!(events[1].get_content(), "Content\n");
        assert_eq!(events[2].part(), TagParts::End);
        assert_eq!(events[2].tag_handle(), Some(&think_tag));
    }

    #[test]
    fn test_nested_registered_tags() {
        let mut builder = XmlParserBuilder::new();
        let outer = builder.register_tag("think");
        let inner = builder.register_tag("inner");
        let parser = builder.build();

        let text = "<think>hi<inner>there</inner>end</think>";
        let events = parser.parse_complete(text);

        assert_eq!(events.len(), 6);
        assert_eq!(events[0].part(), TagParts::Start);
        assert_eq!(events[0].tag_handle(), Some(&outer));
        assert_eq!(events[1].part(), TagParts::Start);
        assert_eq!(events[1].tag_handle(), Some(&inner));
        assert_eq!(events[2].part(), TagParts::Content);
        assert_eq!(events[2].tag_handle(), Some(&inner));
        assert_eq!(events[2].get_content(), "there\n");
        assert_eq!(events[3].part(), TagParts::End);
        assert_eq!(events[3].tag_handle(), Some(&inner));
        assert_eq!(events[4].part(), TagParts::Content);
        assert_eq!(events[4].tag_handle(), Some(&outer));
        assert_eq!(events[4].get_content(), "hiend\n");
        assert_eq!(events[5].part(), TagParts::End);
        assert_eq!(events[5].tag_handle(), Some(&outer));
    }

    #[test]
    fn test_parse_token_equivalent_to_complete() {
        let mut builder = XmlParserBuilder::new();
        builder.register_tag("think");
        let parser = builder.build();

        let text = "<think>Hello</think>";
        let expected = parser.parse_complete(text);

        parser.reset();
        let mut actual = parser.parse_token(text);
        actual.extend(parser.flush());

        assert_eq!(expected, actual);
    }
}
