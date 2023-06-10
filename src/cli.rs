use std::fmt::Display;

#[macro_export]
/// Creates an enum with the given name and variants. The enum will have a
/// method `Self::all() -> &'static [Self]` which returns a slice containing
/// every variant, once, in the order they were defined. The enum will also
/// implement std::fmt::Display, using the provided string representations for
/// each variant.
macro_rules! choices_enum {
    (
        $visibility: vis enum $enum_name: ident {
            $($variant_name: ident as $variant_display: literal),+$(,)?
        }
    ) => {
        $visibility enum $enum_name {
            $(
                $variant_name
            ),+
        }

        impl $enum_name {
            /// Returns a slice of all the variants in this enum
            pub fn all() -> &'static [Self] {
                &[$(
                    Self::$variant_name
                ),+]
            }
        }

        impl Display for $enum_name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match self {
                    $(
                        Self::$variant_name => write!(f, $variant_display)
                    ),+
                }
            }
        }
    };
}

/// Interactively prompts the user to choose one of the provided options
///
/// # Arguments
/// - `prompt_message` - A message to display before asking the user for their
/// choice
/// - `options` - The user's options to choose from
pub fn prompt_choice<'options, T: Display>(
    prompt_message: &str,
    options: &'options [T],
) -> Option<&'options T> {
    loop {
        // Display the prompt message
        println!("{prompt_message}");

        // Display all the user's possible choices
        for (i, option) in options.iter().enumerate() {
            let pad_width = options.len().ilog10() as usize + 1;
            println!("{: >pad_width$}) {option}", i + 1);
        }

        // Get a choice from the user
        match std::io::stdin().lines().next() {
            Some(Ok(line)) => match line.parse::<usize>() {
                Ok(n) if n > 0 && n <= options.len() => {
                    return Some(&options[n - 1]);
                }
                _ => {
                    println!("Invalid choice: '{line}'. Please try again.")
                }
            },
            Some(Err(_)) | None => return None,
        };
        println!();
    }
}
